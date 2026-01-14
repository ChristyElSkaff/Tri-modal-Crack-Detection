import cv2
import json
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO, SAM

# ----------------- CONFIG -----------------
IN_DIR = Path("/home/omenrtx5090/Downloads/pairs/rgb")
OUT_DIR = Path("/home/omenrtx5090/Downloads/pairs/masks_per_object")
YOLO_MODEL = "yolo11x.pt"
SAM2_MODEL = "sam2_b.pt"

CONF = 0.35          # raise to reduce false detections (try 0.35–0.6)
MAX_OBJECTS = 25
DILATE_PIXELS = 3    # 0 disables dilation (use 2-4 if LiDAR is sparse)
# Optional: exclude some classes (like bed/couch/chair) to avoid junk
EXCLUDE_CLASSES = {"bed", "couch", "chair", "sofa"}  # edit as you want
# ------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"
yolo = YOLO(YOLO_MODEL)
sam2 = SAM(SAM2_MODEL)

img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for img_path in sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in img_exts]):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    det = yolo.predict(img, conf=CONF, device=device, verbose=False)[0]
    if det.boxes is None or len(det.boxes) == 0:
        print(f"[{img_path.name}] no detections")
        continue

    boxes = det.boxes.xyxy.detach().cpu().numpy()
    confs = det.boxes.conf.detach().cpu().numpy()
    clss  = det.boxes.cls.detach().cpu().numpy().astype(int)
    names = det.names

    # sort by confidence, keep top N
    order = np.argsort(-confs)[:MAX_OBJECTS]

    meta = []
    obj_id = 0

    for i in order:
        cls_name = names[int(clss[i])]
        if cls_name in EXCLUDE_CLASSES:
            continue

        x1, y1, x2, y2 = boxes[i].tolist()
        c = float(confs[i])

        # SAM2 with a single bbox prompt
        seg = sam2(img, bboxes=[[x1, y1, x2, y2]], verbose=False)[0]
        if seg.masks is None or seg.masks.data is None:
            continue

        m = seg.masks.data[0].detach().cpu().numpy()  # (h,w)
        mask = (m > 0.5).astype(np.uint8) * 255
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if DILATE_PIXELS > 0:
            k = 2 * DILATE_PIXELS + 1
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        out_mask = OUT_DIR / f"{img_path.stem}_obj{obj_id:02d}_{cls_name}_conf{c:.2f}.png"
        cv2.imwrite(str(out_mask), mask)

        meta.append({
            "obj_id": obj_id,
            "mask_file": out_mask.name,
            "class": cls_name,
            "conf": c,
            "bbox_xyxy": [x1, y1, x2, y2],
        })
        obj_id += 1

    # Save metadata for that image
    (OUT_DIR / f"{img_path.stem}_objects.json").write_text(json.dumps(meta, indent=2))
    print(f"[{img_path.name}] saved {len(meta)} object masks")

print("Done.")