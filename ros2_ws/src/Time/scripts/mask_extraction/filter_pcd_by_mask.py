#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2


def load_pcd_xyz(pcd_path: Path) -> np.ndarray:
    """Minimal ASCII PCD loader for x y z (works for common saved PCDs)."""
    with open(pcd_path, "r") as f:
        lines = f.readlines()

    data_idx = None
    fields = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        if s.startswith("DATA"):
            data_idx = i + 1
            break
    if data_idx is None or fields is None:
        raise RuntimeError(f"Could not parse PCD header: {pcd_path}")

    try:
        ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError:
        raise RuntimeError(f"PCD missing x/y/z fields: {fields}")

    pts = []
    for line in lines[data_idx:]:
        if not line.strip():
            continue
        vals = line.split()
        pts.append([float(vals[ix]), float(vals[iy]), float(vals[iz])])

    return np.asarray(pts, dtype=np.float32)


def save_pcd_xyz_ascii(pcd_path: Path, xyz: np.ndarray):
    xyz = xyz.astype(np.float32)
    n = int(xyz.shape[0])
    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii"
    ]) + "\n"
    with open(pcd_path, "w") as f:
        f.write(header)
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, type=str)
    ap.add_argument("--mask", required=True, type=str)
    ap.add_argument("--pcd", required=True, type=str)
    ap.add_argument("--out_pcd", required=True, type=str)

    # intrinsics
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)

    # extrinsic: lidar->camera as translation + RPY (radians)
    ap.add_argument("--tx", type=float, required=True)
    ap.add_argument("--ty", type=float, required=True)
    ap.add_argument("--tz", type=float, required=True)
    ap.add_argument("--roll", type=float, required=True)
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--yaw", type=float, required=True)

    # debug overlay
    ap.add_argument("--debug_png", default="", help="save debug overlay png")
    ap.add_argument("--dot", type=int, default=2)

    args = ap.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None:
        raise FileNotFoundError("Could not read rgb or mask.")

    H_img, W_img = rgb.shape[:2]

    # Ensure mask matches image size
    if mask.shape[0] != H_img or mask.shape[1] != W_img:
        mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    pts_l = load_pcd_xyz(Path(args.pcd))  # (N,3)

    # Build R from roll/pitch/yaw (ZYX: yaw->pitch->roll)
    cr, sr = np.cos(args.roll), np.sin(args.roll)
    cp, sp = np.cos(args.pitch), np.sin(args.pitch)
    cy, sy = np.cos(args.yaw), np.sin(args.yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    t = np.array([args.tx, args.ty, args.tz], dtype=np.float32)

    # lidar -> camera
    pts_c = (R @ pts_l.T).T + t

    # keep points in front of camera (positive Z in THIS camera model)
    z = pts_c[:, 2]
    valid_z = z > 1e-6
    pts_c = pts_c[valid_z]
    pts_l_valid = pts_l[valid_z]

    # project
    u = (args.fx * (pts_c[:, 0] / pts_c[:, 2]) + args.cx)
    v = (args.fy * (pts_c[:, 1] / pts_c[:, 2]) + args.cy)

    ui = u.astype(np.int32)
    vi = v.astype(np.int32)

    inside = (ui >= 0) & (ui < W_img) & (vi >= 0) & (vi < H_img)

    ui_in = ui[inside]
    vi_in = vi[inside]
    pts_in = pts_l_valid[inside]

    # mask filter (white pixels)
    mvals = mask[vi_in, ui_in]
    keep = mvals > 127
    out = pts_in[keep]

    # DEBUG: show projected points colored by keep/reject
    if args.debug_png:
        vis = rgb.copy()

        # tint mask area
        mask_bin = (mask > 127)
        vis[mask_bin] = (0.7 * vis[mask_bin] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)

        # draw points
        for x, y, k in zip(ui_in, vi_in, keep):
            color = (0, 255, 0) if k else (0, 0, 255)  # green kept, red rejected
            cv2.circle(vis, (int(x), int(y)), int(args.dot), color, -1)

        cv2.imwrite(args.debug_png, vis)
        print(f"Saved debug overlay: {args.debug_png}")

    save_pcd_xyz_ascii(Path(args.out_pcd), out)
    print(f"Saved {args.out_pcd} with {len(out)} points (kept from {len(pts_l)})")


if __name__ == "__main__":
    main()
