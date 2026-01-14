#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

PAIRS_DIR = Path("/home/omenrtx5090/Downloads/pairs")

RGB_DIR   = PAIRS_DIR / "rgb"
PCD_DIR   = PAIRS_DIR / "lidar"
MASK_DIR  = PAIRS_DIR / "masks" / "union_masks"
OUT_DIR   = PAIRS_DIR / "objects_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- intrinsics (yours) ----
FX, FY = 3386.3, 3386.0
CX, CY = 846.188, 406.173

# ---- extrinsic lidar->camera (yours from static_transform_publisher) ----
TX, TY, TZ = -0.05, 0.0, -0.05
ROLL, PITCH, YAW= -4.76, 3.18, -4.08

FILTER_SCRIPT = Path("filter_pcd_by_mask.py")  # make sure it’s in the same folder

img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_matching_pcd(img_path: Path) -> Path:
    # Try common naming patterns:
    # rgb_000123.png  -> cloud_000123.pcd
    stem = img_path.stem  # e.g. rgb_000123
    idx = stem.split("_")[-1]

    candidates = [
        PCD_DIR / f"cloud_{idx}.pcd",
        PCD_DIR / f"pcd_{idx}.pcd",
        PCD_DIR / f"{stem}.pcd",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def main():
    images = sorted([p for p in RGB_DIR.iterdir() if p.suffix.lower() in img_exts])
    if not images:
        print(f"No images found in {RGB_DIR}")
        sys.exit(1)

    ok, skipped = 0, 0
    for img in images:
        stem = img.stem  # rgb_000123
        mask = MASK_DIR / f"{stem}_ALL_mask.png"
        if not mask.exists():
            print(f"[SKIP] no mask: {mask.name}")
            skipped += 1
            continue

        pcd = find_matching_pcd(img)
        if pcd is None:
            print(f"[SKIP] no matching pcd for: {img.name}")
            skipped += 1
            continue

        out_pcd = OUT_DIR / f"{stem}_objects_only.pcd"

        cmd = [
            sys.executable, str(FILTER_SCRIPT),
            "--rgb", str(img),
            "--mask", str(mask),
            "--pcd", str(pcd),
            "--out_pcd", str(out_pcd),
            "--fx", str(FX), "--fy", str(FY), "--cx", str(CX), "--cy", str(CY),
            "--tx", str(TX), "--ty", str(TY), "--tz", str(TZ),
            "--roll", str(ROLL), "--pitch", str(PITCH), "--yaw", str(YAW),
        ]

        print(f"[RUN] {img.name} + {pcd.name} -> {out_pcd.name}")
        subprocess.run(cmd, check=True)
        ok += 1

    print(f"Done. Processed: {ok}, skipped: {skipped}")
    print(f"Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()