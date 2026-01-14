#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2


def load_pcd_xyz(pcd_path: Path) -> np.ndarray:
    """
    Load x,y,z from a PCD file (supports DATA ascii and DATA binary).
    Does NOT support DATA binary_compressed.
    """
    import re

    with open(pcd_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PCD header.")
            s = line.decode("ascii", errors="ignore").strip()
            header_lines.append(s)
            if s.startswith("DATA"):
                data_type = s.split()[1].lower()
                break

        # Parse header fields
        fields = sizes = types = counts = None
        points = None

        for s in header_lines:
            if s.startswith("FIELDS"):
                fields = s.split()[1:]
            elif s.startswith("SIZE"):
                sizes = list(map(int, s.split()[1:]))
            elif s.startswith("TYPE"):
                types = s.split()[1:]
            elif s.startswith("COUNT"):
                counts = list(map(int, s.split()[1:]))
            elif s.startswith("POINTS"):
                points = int(s.split()[1])

        if fields is None or sizes is None or types is None:
            raise RuntimeError(f"PCD header missing FIELDS/SIZE/TYPE: {pcd_path}")

        if counts is None:
            counts = [1] * len(fields)

        if points is None:
            # fallback: WIDTH*HEIGHT if POINTS missing
            width = height = None
            for s in header_lines:
                if s.startswith("WIDTH"):
                    width = int(s.split()[1])
                elif s.startswith("HEIGHT"):
                    height = int(s.split()[1])
            if width is None or height is None:
                raise RuntimeError("PCD header missing POINTS and WIDTH/HEIGHT.")
            points = width * height

        # Build numpy dtype for one point
        np_fields = []
        for name, sz, tp, ct in zip(fields, sizes, types, counts):
            if tp == "F":
                base = np.float32 if sz == 4 else np.float64
            elif tp == "I":
                base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[sz]
            elif tp == "U":
                base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[sz]
            else:
                raise RuntimeError(f"Unsupported PCD TYPE {tp}")

            if ct == 1:
                np_fields.append((name, base))
            else:
                # expand multi-count fields: name_0, name_1, ...
                for k in range(ct):
                    np_fields.append((f"{name}_{k}", base))

        dtype = np.dtype(np_fields)

        if data_type == "ascii":
            # Read remaining as text
            text = f.read().decode("ascii", errors="ignore").strip().splitlines()
            pts = []
            # locate x y z columns from expanded dtype
            # (common case: fields contain x y z with count=1)
            try:
                ix = list(dtype.names).index("x")
                iy = list(dtype.names).index("y")
                iz = list(dtype.names).index("z")
            except ValueError:
                raise RuntimeError(f"PCD does not contain x/y/z in FIELDS: {fields}")

            for line in text:
                if not line.strip():
                    continue
                vals = line.split()
                pts.append([float(vals[ix]), float(vals[iy]), float(vals[iz])])
            return np.asarray(pts, dtype=np.float32)

        elif data_type == "binary":
            # Read binary blob and interpret with dtype
            blob = f.read()
            arr = np.frombuffer(blob, dtype=dtype, count=points)

            # x,y,z exist?
            if "x" not in arr.dtype.names or "y" not in arr.dtype.names or "z" not in arr.dtype.names:
                raise RuntimeError(f"PCD does not contain x/y/z in FIELDS: {fields}")

            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
            return xyz

        elif data_type == "binary_compressed":
            raise RuntimeError("PCD DATA binary_compressed not supported. Convert to ascii or binary first.")
        else:
            raise RuntimeError(f"Unknown PCD DATA type: {data_type}")



def rpy_to_R(roll, pitch, yaw) -> np.ndarray:
    """ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float32)
    return Rz @ Ry @ Rx


def build_projected_mask(points_lidar: np.ndarray,
                         H: int, W: int,
                         fx: float, fy: float, cx: float, cy: float,
                         tx: float, ty: float, tz: float,
                         roll: float, pitch: float, yaw: float,
                         zbuffer: bool = True) -> np.ndarray:
    """
    Returns a binary image (H,W) where pixels hit by projected points are True.
    Uses optional z-buffer to keep only closest point per pixel (reduces leakage).
    """
    if points_lidar.size == 0:
        return np.zeros((H, W), dtype=bool)

    R = rpy_to_R(roll, pitch, yaw)
    t = np.array([tx, ty, tz], dtype=np.float32)

    # LiDAR -> Camera
    pc = (R @ points_lidar.T).T + t

    Z = pc[:, 2]
    valid = np.isfinite(pc).all(axis=1) & (Z > 1e-6)
    pc = pc[valid]
    if pc.shape[0] == 0:
        return np.zeros((H, W), dtype=bool)

    X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    finite = np.isfinite(u) & np.isfinite(v)
    u = u[finite]
    v = v[finite]
    Z = Z[finite]

    ui = np.rint(u).astype(np.int32)  # round to nearest pixel
    vi = np.rint(v).astype(np.int32)

    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[inside]
    vi = vi[inside]
    Z = Z[inside]

    proj = np.zeros((H, W), dtype=bool)
    if ui.size == 0:
        return proj

    if not zbuffer:
        proj[vi, ui] = True
        return proj

    # Z-buffer: for each pixel keep the smallest Z (closest)
    # Use a flat index for fast grouping
    flat = vi.astype(np.int64) * W + ui.astype(np.int64)
    order = np.lexsort((Z, flat))  # sort by flat then Z
    flat_s = flat[order]
    vi_s = vi[order]
    ui_s = ui[order]
    Z_s = Z[order]

    # pick first occurrence for each flat (since sorted by Z within flat)
    first = np.ones_like(flat_s, dtype=bool)
    first[1:] = flat_s[1:] != flat_s[:-1]

    proj[vi_s[first], ui_s[first]] = True
    return proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=False, default="", help="Optional. Only used to infer H,W if mask not used.")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--pcd", required=True)

    # Intrinsics defaults (your FLIR intrinsics)
    ap.add_argument("--fx", type=float, default=3386.3)
    ap.add_argument("--fy", type=float, default=3386.0)
    ap.add_argument("--cx", type=float, default=846.188)
    ap.add_argument("--cy", type=float, default=406.173)

    # Extrinsic defaults (your static_transform_publisher lidar->camera)
    ap.add_argument("--tx", type=float, default=-0.05)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=-0.05)
    ap.add_argument("--roll", type=float, default=-4.69)
    ap.add_argument("--pitch", type=float, default=3.18)
    ap.add_argument("--yaw", type=float, default=-4.08)

    ap.add_argument("--no_zbuffer", action="store_true", help="Disable closest-point-per-pixel.")
    ap.add_argument("--dilate", type=int, default=0, help="Dilate projected mask by this many pixels (robustness).")
    ap.add_argument("--save_proj_png", default="", help="Save projected mask PNG for debugging.")
    ap.add_argument("--save_overlay_png", default="", help="Save overlay (mask + projected) for debugging.")

    args = ap.parse_args()

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {args.mask}")
    H, W = mask.shape[:2]
    mask_bool = mask > 127

    pts_l = load_pcd_xyz(Path(args.pcd))
    proj_bool = build_projected_mask(
        pts_l, H, W,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        tx=args.tx, ty=args.ty, tz=args.tz,
        roll=args.roll, pitch=args.pitch, yaw=args.yaw,
        zbuffer=(not args.no_zbuffer),
    )

    if args.dilate > 0:
        k = 2 * args.dilate + 1
        kernel = np.ones((k, k), np.uint8)
        proj_u8 = proj_bool.astype(np.uint8) * 255
        proj_u8 = cv2.dilate(proj_u8, kernel, iterations=1)
        proj_bool = proj_u8 > 0

    inter = np.logical_and(proj_bool, mask_bool).sum()
    union = np.logical_or(proj_bool, mask_bool).sum()

    proj_sum = proj_bool.sum()
    mask_sum = mask_bool.sum()

    iou = (inter / union) if union > 0 else 0.0
    precision = (inter / proj_sum) if proj_sum > 0 else 0.0
    recall = (inter / mask_sum) if mask_sum > 0 else 0.0

    print("=== Reprojection Metrics ===")
    print(f"PCD points: {len(pts_l)}")
    print(f"Mask white pixels: {mask_sum}")
    print(f"Projected pixels: {proj_sum}")
    print(f"Intersection: {inter}")
    print(f"Union: {union}")
    print(f"IoU: {iou:.4f}")
    print(f"Precision (proj->mask): {precision:.4f}")
    print(f"Recall (mask coverage): {recall:.4f}")

    if args.save_proj_png:
        out = (proj_bool.astype(np.uint8) * 255)
        cv2.imwrite(args.save_proj_png, out)
        print(f"Saved projected mask: {args.save_proj_png}")

    if args.save_overlay_png:
        # overlay: mask = green, projected = red, intersection = yellow
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[mask_bool] = (0, 255, 0)
        overlay[proj_bool] = (0, 0, 255)
        overlay[np.logical_and(mask_bool, proj_bool)] = (0, 255, 255)
        cv2.imwrite(args.save_overlay_png, overlay)
        print(f"Saved overlay: {args.save_overlay_png}")


if __name__ == "__main__":
    main()