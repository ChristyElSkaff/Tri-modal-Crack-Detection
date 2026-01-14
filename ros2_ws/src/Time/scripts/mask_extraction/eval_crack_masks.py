#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO
from pathlib import Path

def metrics(gt_bool: np.ndarray, pr_bool: np.ndarray):
    gt = gt_bool.astype(bool)
    pr = pr_bool.astype(bool)

    tp = np.logical_and(gt, pr).sum()
    fp = np.logical_and(~gt, pr).sum()
    fn = np.logical_and(gt, ~pr).sum()

    # If both empty => perfect match convention
    if gt.sum() == 0 and pr.sum() == 0:
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0,
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"iou": float(iou), "dice": float(dice),
            "precision": float(precision), "recall": float(recall),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--pred_mask", required=True, help="Predicted mask PNG (binary 0/255 or grayscale).")
    ap.add_argument("--image_name", default="flir_000002.png", help="COCO 'file_name' to evaluate.")
    ap.add_argument("--gt_category_name", default="crack")
    ap.add_argument("--pred_thresh", type=int, default=127)
    ap.add_argument("--height", type=int, default=0, help="Override H if JSON missing it (e.g., 1080).")
    ap.add_argument("--width", type=int, default=0, help="Override W if JSON missing it (e.g., 1440).")
    ap.add_argument("--resize_pred", action="store_true", help="Resize pred to GT size if needed.")
    args = ap.parse_args()

    coco = COCO(args.coco_json)

    # ---- find category id ----
    cat_ids = coco.getCatIds(catNms=[args.gt_category_name])
    if not cat_ids:
        cats = [c["name"] for c in coco.loadCats(coco.getCatIds())]
        raise RuntimeError(f"Category '{args.gt_category_name}' not found. Available: {cats}")
    crack_cat = cat_ids[0]

    # ---- find image id by file_name ----
    

    img_id = None
    img_info = None

    for im in coco.loadImgs(coco.getImgIds()):
        fn = str(im.get("file_name", ""))
        base = Path(fn).name  # e.g. "7d4ed5cf-flir_000000.png"

    # Accept:
    # - exact match: "flir_000000.png"
    # - suffix match: "...-flir_000000.png"
        if base == args.image_name or base.endswith(args.image_name):
            img_id = im["id"]
            img_info = im
            break

    if img_id is None:
    # helpful debug print
        examples = [Path(i.get("file_name", "")).name for i in coco.loadImgs(coco.getImgIds())[:10]]
        raise RuntimeError(
            f"Image '{args.image_name}' not found. Example image names in JSON: {examples}"
        )


    H = int(img_info.get("height") or 0)
    W = int(img_info.get("width") or 0)

    # If height/width missing in JSON, allow override
    if (H == 0 or W == 0):
        if args.height > 0 and args.width > 0:
            H, W = args.height, args.width
        else:
            raise RuntimeError(
                "COCO JSON is missing image height/width. "
                "Run again with --height 1080 --width 1440 (for your FLIR)."
            )

    # ---- build GT mask from COCO annotations ----
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[crack_cat])
    anns = coco.loadAnns(ann_ids)

    gt = np.zeros((H, W), dtype=np.uint8)
    for ann in anns:
        gt = np.maximum(gt, coco.annToMask(ann).astype(np.uint8) * 255)
    gt_bool = gt > 127

    # ---- load prediction mask ----
    pr = cv2.imread(args.pred_mask, cv2.IMREAD_GRAYSCALE)
    if pr is None:
        raise FileNotFoundError(f"Could not read predicted mask: {args.pred_mask}")

    if pr.shape != (H, W):
        if args.resize_pred:
            pr = cv2.resize(pr, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            raise RuntimeError(f"Pred mask size {pr.shape} != GT size {(H, W)}. Use --resize_pred.")

    pr_bool = pr > args.pred_thresh

    m = metrics(gt_bool, pr_bool)

    print("\n=== Single-image evaluation ===")
    print(f"Image: {args.image_name}")
    print(f"GT crack pixels : {int(gt_bool.sum())}")
    print(f"Pred crack pixels: {int(pr_bool.sum())}")
    print(f"IoU     : {m['iou']:.4f}")
    print(f"Dice    : {m['dice']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall  : {m['recall']:.4f}")
    print(f"TP/FP/FN: {m['tp']} / {m['fp']} / {m['fn']}")


if __name__ == "__main__":
    main()
