import cv2
import numpy as np
from pathlib import Path
import torch

from ultralytics import YOLO, SAM  # SAM 2 supported in Ultralytics

# ----------------- CONFIG -----------------
IN_DIR = Path("/home/omenrtx5090/Downloads/pairs/rgb")
OUT_DIR = Path("/home/omenrtx5090/Downloads/pairs/masks")
YOLO_MODEL = "yolo11x.pt"     # most accurate detector (heavier); use yolo11l.pt/yolo11m.pt if too slow
SAM2_MODEL = "sam2_b.pt"      # good default SAM2 checkpoint in Ultralytics docs

CONF = 0.25                   # raise to reduce false detections (e.g. 0.4)
MAX_OBJECTS = 50              # safety limit
DILATE_PIXELS = 3             # helps catch LiDAR points near edges (0 disables)

# --- NEW: classes you do NOT want in the mask ---
UNWANTED_CLASSES = {"bed"}    # add more: {"bed", "couch", "tv", ...}

# --- OPTIONAL: keep only the most central object (helpful if room objects distract) ---
ONLY_KEEP_CENTER_OBJECT = False
# ------------------------------------------

(OUT_DIR / "union_masks").mkdir(parents=True, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"
yolo = YOLO(YOLO_MODEL)
sam2 = SAM(SAM2_MODEL)

img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for img_path in sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in img_exts]):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    # 1) YOLO detect -> boxes
    det = yolo.predict(img, conf=CONF, device=device, verbose=False)[0]
    if det.boxes is None or len(det.boxes) == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        cv2.imwrite(str(OUT_DIR / "union_masks" / f"{img_path.stem}_ALL_mask.png"), empty)
        print(f"[{img_path.name}] no detections -> saved empty union mask")
        continue

    boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()  # (N,4)
    confs = det.boxes.conf.detach().cpu().numpy()       # (N,)
    clss = det.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)

    # class names dict: {id: "name"}
    names = det.names

    # ---------------- NEW: filter out unwanted classes ----------------
    keep = []
    for i in range(len(boxes_xyxy)):
        cls_id = int(clss[i])
        cls_name = names.get(cls_id, str(cls_id))
        if cls_name in UNWANTED_CLASSES:
            continue
        keep.append(i)

    if len(keep) == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        out_path = OUT_DIR / "union_masks" / f"{img_path.stem}_ALL_mask.png"
        cv2.imwrite(str(out_path), empty)
        print(f"[{img_path.name}] all detections were filtered (e.g. only unwanted classes) -> empty mask")
        continue

    boxes_xyxy = boxes_xyxy[keep]
    confs = confs[keep]
    clss = clss[keep]

    # keep top-N by confidence (after filtering)
    order = np.argsort(-confs)[:MAX_OBJECTS]
    boxes_xyxy = boxes_xyxy[order]
    confs = confs[order]
    clss = clss[order]

    # OPTIONAL: keep only the most central detection
    if ONLY_KEEP_CENTER_OBJECT and len(boxes_xyxy) > 1:
        cx0, cy0 = W / 2.0, H / 2.0
        centers = np.column_stack(((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0,
                                   (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0))
        d2 = (centers[:, 0] - cx0) ** 2 + (centers[:, 1] - cy0) ** 2
        best = int(np.argmin(d2))
        boxes_xyxy = boxes_xyxy[[best]]
        confs = confs[[best]]
        clss = clss[[best]]

    # Debug print: what you kept
    kept_labels = [names.get(int(c), str(int(c))) for c in clss]
    print(f"[{img_path.name}] kept {len(boxes_xyxy)} dets:", list(zip(kept_labels, confs.round(3))))

    bboxes = boxes_xyxy.tolist()  # [[x1,y1,x2,y2], ...]

    # 2) SAM2 segmentation with YOLO boxes as prompts
    seg = sam2(img, bboxes=bboxes, verbose=False)[0]
    if seg.masks is None or seg.masks.data is None:
        empty = np.zeros((H, W), dtype=np.uint8)
        cv2.imwrite(str(OUT_DIR / "union_masks" / f"{img_path.stem}_ALL_mask.png"), empty)
        print(f"[{img_path.name}] SAM2 returned no masks -> saved empty union mask")
        continue

    masks = seg.masks.data.detach().cpu().numpy()  # (N, h, w)

    # union mask (all kept objects)
    union = (masks.max(axis=0) > 0.5).astype(np.uint8) * 255
    if union.shape != (H, W):
        union = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST)

    # optional dilation (helps with sparse lidar + small calibration errors)
    if DILATE_PIXELS > 0:
        k = 2 * DILATE_PIXELS + 1
        kernel = np.ones((k, k), np.uint8)
        union = cv2.dilate(union, kernel, iterations=1)

    out_path = OUT_DIR / "union_masks" / f"{img_path.stem}_ALL_mask.png"
    cv2.imwrite(str(out_path), union)
    print(f"[{img_path.name}] saved union mask -> {out_path}")

print("Done.")
