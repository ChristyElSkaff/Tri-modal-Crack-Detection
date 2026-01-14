import numpy as np
import cv2
from pathlib import Path

# -------------------- INPUTS --------------------
img_path = Path("/home/omenrtx5090/Downloads/pairs/rgb/rgb_000001.png")
pcd_path = Path("/home/omenrtx5090/Downloads/pairs/objects_only/rgb_000001_objects_only.pcd")  # or your full cloud

# Camera intrinsics
K = np.array([[3386.3,   0.0, 846.188],
              [  0.0, 3386.0, 406.173],
              [  0.0,   0.0,   1.0  ]], dtype=np.float64)

# Distortion: [k1,k2,p1,p2,k3] (use zeros if your image is rectified)
dist = np.array([-0.3341, 0.6372, 0.0, 0.0, 0.0], dtype=np.float64)

# Your TF: livox_frame -> flir_camera
tx, ty, tz = -0.05, 0.0, -0.05
roll, pitch, yaw = -4.76, 3.18, -4.08  # radians

SAMPLE_N = 4000  # reduce clutter
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
    with open(path, "r") as f:
        lines = f.readlines()
    fields = None
    data_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        if s.startswith("DATA"):
            if "ascii" not in s:
                raise ValueError("Expected ASCII PCD.")
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


def invert_rt(R, t):
    Rinv = R.T
    tinv = -Rinv @ t
    return Rinv, tinv


def apply_optical(R, t):
    """
    Convert a 'camera' frame to 'optical' frame using the common ROS convention.
    This is the usual fixed rotation between camera_link and camera_optical_frame.
    """
    R_cam_to_opt = np.array([[0, -1, 0],
                             [0,  0, -1],
                             [1,  0, 0]], dtype=np.float64)
    R2 = R_cam_to_opt @ R
    t2 = R_cam_to_opt @ t
    return R2, t2


def project_points(img, pts_L, R_C_L, t_C_L, tag):
    H, W = img.shape[:2]
    # Transform LiDAR -> Camera
    pts_C = (R_C_L @ pts_L.T).T + t_C_L
    front = pts_C[:, 2] > 1e-6
    pts_C = pts_C[front]
    if len(pts_C) == 0:
        out = img.copy()
        cv2.putText(out, f"{tag}: no points in front", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return out

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.projectPoints(pts_C.astype(np.float64), rvec, tvec, K, dist)
    uv = uv.reshape(-1, 2)

    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    uv = uv[finite]
    u = uv[:, 0].astype(np.int32)
    v = uv[:, 1].astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inside]; v = v[inside]

    out = img.copy()
    # draw points
    for x, y in zip(u, v):
        out[y, x] = (0, 255, 0)
    cv2.putText(out, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(out, f"drawn: {len(u)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return out


# Load data
img = cv2.imread(str(img_path))
assert img is not None, "Image not found"
pts = read_pcd_ascii_xyz(pcd_path)

# sample for readability
if len(pts) > SAMPLE_N:
    idx = np.random.choice(len(pts), SAMPLE_N, replace=False)
    pts = pts[idx]

R0 = rot_zyx(yaw, pitch, roll)
t0 = np.array([tx, ty, tz], dtype=np.float64)

# A) forward
A = project_points(img, pts, R0, t0, "A forward (lidar->cam)")
# B) inverted
Rinv, tinv = invert_rt(R0, t0)
B = project_points(img, pts, Rinv, tinv, "B inverted")
# C) forward + optical
Ropt, topt = apply_optical(R0, t0)
C = project_points(img, pts, Ropt, topt, "C forward+optical")
# D) inverted + optical
Ropt2, topt2 = apply_optical(Rinv, tinv)
D = project_points(img, pts, Ropt2, topt2, "D inverted+optical")

out_dir = Path("debug_overlays")
out_dir.mkdir(exist_ok=True)

cv2.imwrite(str(out_dir / "A_forward.png"), A)
cv2.imwrite(str(out_dir / "B_inverted.png"), B)
cv2.imwrite(str(out_dir / "C_forward_optical.png"), C)
cv2.imwrite(str(out_dir / "D_inverted_optical.png"), D)

print("Wrote overlays to:", out_dir.resolve())