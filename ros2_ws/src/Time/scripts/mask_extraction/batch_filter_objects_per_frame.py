from pathlib import Path
import subprocess
import sys

PAIRS_DIR = Path("/home/omenrtx5090/Downloads/pairs")

RGB_DIR  = PAIRS_DIR / "rgb"
PCD_DIR  = PAIRS_DIR / "lidar"  # change to pcd_ascii if you converted them
MASK_DIR = PAIRS_DIR / "masks_per_object"
OUT_DIR  = PAIRS_DIR / "objects_only_per_object"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILTER_SCRIPT = Path("filter_pcd_by_mask.py")  # the script that outputs objects_only pcd

# intrinsics (yours)
FX, FY = 3386.3, 3386.0
CX, CY = 846.188, 406.173

# extrinsic lidar->camera (yours)
TX, TY, TZ = -0.05, 0.0, -0.05
ROLL, PITCH, YAW = -4.76, 3.18, -4.08

img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_pcd_for_rgb(rgb_path: Path) -> Path:
    # rgb_000001.png -> try common pcd naming
    idx = rgb_path.stem.split("_")[-1]
    for cand in [
        PCD_DIR / f"cloud_{idx}.pcd",
        PCD_DIR / f"{rgb_path.stem}.pcd",
        PCD_DIR / f"pcd_{idx}.pcd",
    ]:
        if cand.exists():
            return cand
    return None

for rgb in sorted([p for p in RGB_DIR.iterdir() if p.suffix.lower() in img_exts]):
    pcd = find_pcd_for_rgb(rgb)
    if pcd is None:
        print(f"[SKIP] no PCD for {rgb.name}")
        continue

    # all masks for that frame: rgb_000001_objXX_*.png
    masks = sorted(MASK_DIR.glob(f"{rgb.stem}_obj*_*.png"))
    if not masks:
        print(f"[{rgb.name}] no object masks")
        continue

    for mask_path in masks:
        # output file keeps same name as mask for clarity
        out_pcd = OUT_DIR / (mask_path.stem + ".pcd")

        cmd = [
            sys.executable, str(FILTER_SCRIPT),
            "--rgb", str(rgb),
            "--mask", str(mask_path),
            "--pcd", str(pcd),
            "--out_pcd", str(out_pcd),
            "--fx", str(FX), "--fy", str(FY), "--cx", str(CX), "--cy", str(CY),
            "--tx", str(TX), "--ty", str(TY), "--tz", str(TZ),
            "--roll", str(ROLL), "--pitch", str(PITCH), "--yaw", str(YAW),
        ]
        subprocess.run(cmd, check=True)
        print(f"  -> {out_pcd.name}")

print("Done.")