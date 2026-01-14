import os
from pathlib import Path
import numpy as np
import cv2

# ------------------ EDIT THESE ------------------
SESSION = Path("/home/omenrtx5090/Downloads/pairs")
RGB_DIR   = SESSION / "rgb"
PCD_DIR   = SESSION / "lidar"
MASK_DIR  = SESSION / "masks"
OUT_DIR   = SESSION / "object_clouds"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Intrinsics
K = np.array([[3386.3,   0.0, 846.188],
              [  0.0, 3386.0, 406.173],
              [  0.0,   0.0,   1.0  ]], dtype=np.float64)

# Distortion (set to zeros if your image is already rectified)
dist = np.array([-0.3341, 0.6372, 0.0, 0.0, 0.0], dtype=np.float64)

# TF: livox_frame -> flir_camera (your forward transform)
tx, ty, tz = -0.05, 0.0, -0.05
yaw, pitch, roll = -4.69, 3.18, -4.08  # radians

# Optional: shrink mask edges to reduce noise
ERODE_PIXELS = 1   # try 0,1,2
# ------------------------------------------------

def rot_zyx(yaw, pitch, roll):
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],
                   [ 0, 1,  0],
                   [-sy,0, cy]], dtype=np.float64)
    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float64)
    return Rz @ Ry @ Rx

def read_pcd_ascii_xyz(path: Path):
    # ASCII PCD reader for x,y,z
    with open(path, "r") as f:
        lines = f.readlines()
    fields = None
    data_start = None
    points = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        if s.startswith("DATA"):
            if "ascii" not in s:
                raise ValueError(f"{path} is not ASCII PCD. Convert to ASCII or use Open3D.")
            data_start = i + 1
            break
    if fields is None or data_start is None:
        raise ValueError("Bad PCD header.")
    data_lines = [ln.strip() for ln in lines[data_start:] if ln.strip()]
    data = np.loadtxt(data_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    return data[:, [ix, iy, iz]]

def write_pcd_ascii_xyz(path: Path, xyz: np.ndarray):
    xyz = xyz.astype(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {xyz.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {xyz.shape[0]}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def main():
    R = rot_zyx(yaw, pitch, roll)
    t = np.array([tx, ty, tz], dtype=np.float64)

    rgb_files = sorted(RGB_DIR.glob("*.png"))
    if not rgb_files:
        raise RuntimeError(f"No pngs found in {RGB_DIR}")

    for rgb_path in rgb_files:
        stem = rgb_path.stem  # rgb_000000
        idx = stem.split("_")[-1]  # 000000

        pcd_path  = PCD_DIR  / f"cloud_{idx}.pcd"
        mask_path = MASK_DIR / f"mask_{idx}.png"

        if not pcd_path.exists():
            print(f"[SKIP] Missing PCD: {pcd_path}")
            continue
        if not mask_path.exists():
            print(f"[SKIP] Missing mask: {mask_path}")
            continue

        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"[SKIP] Failed to read {rgb_path} or {mask_path}")
            continue

        H, W = mask.shape[:2]
        if img.shape[0] != H or img.shape[1] != W:
            print(f"[WARN] Size mismatch, resizing mask to image for {idx}")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            H, W = mask.shape[:2]

        if ERODE_PIXELS > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ERODE_PIXELS+1, 2*ERODE_PIXELS+1))
            mask = cv2.erode(mask, k, iterations=1)

        pts_L = read_pcd_ascii_xyz(pcd_path)  # (N,3)

        # LiDAR -> Camera
        pts_C = (R @ pts_L.T).T + t

        # keep points in front of camera
        front = pts_C[:, 2] > 1e-6
        pts_C = pts_C[front]
        pts_L = pts_L[front]
        if len(pts_C) == 0:
            print(f"[{idx}] no points in front")
            continue

        # Project with OpenCV (handles distortion)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        uv, _ = cv2.projectPoints(pts_C.astype(np.float64), rvec, tvec, K, dist)
        uv = uv.reshape(-1, 2)

        finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        uv = uv[finite]
        pts_L = pts_L[finite]

        u = uv[:, 0].astype(np.int32)
        v = uv[:, 1].astype(np.int32)

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[inside]; v = v[inside]
        pts_L = pts_L[inside]

        # Mask filter
        keep = mask[v, u] > 0
        obj_pts = pts_L[keep]

        out_pcd = OUT_DIR / f"object_{idx}.pcd"
        write_pcd_ascii_xyz(out_pcd, obj_pts)

        print(f"[{idx}] kept {len(obj_pts)} points -> {out_pcd.name}")

if __name__ == "__main__":
    main()