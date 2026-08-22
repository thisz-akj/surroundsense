"""
Runs a pretrained YOLO11-seg (nano, COCO-trained) model on the 50 stitched
surround-view images, to see whether a standard instance-segmentation model
-- trained entirely on normal, eye-level/dashcam-style photos -- can still
find and mask objects once they've been warped into this top-down,
fisheye-remapped, multi-camera-blended representation.

Saves annotated images (boxes + masks + class labels) plus a per-image
JSON summary of what was detected.
"""

import json
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import EXPERIMENTS_DIR, YOLO_SEG_MODEL_PATH  # noqa: E402

IN_DIR = os.path.join(EXPERIMENTS_DIR, "sample50_fv_scaled_079")
OUT_DIR = os.path.join(EXPERIMENTS_DIR, "yolo_seg_50")
MODEL_PATH = YOLO_SEG_MODEL_PATH


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    tags = sorted(
        f[:-4] for f in os.listdir(IN_DIR)
        if f.startswith("idx") and f.endswith(".png")
    )

    summary = {}
    for tag in tags:
        img_path = f"{IN_DIR}/{tag}.png"
        results = model(img_path, verbose=False)
        r = results[0]

        annotated = r.plot()
        import cv2
        cv2.imwrite(f"{OUT_DIR}/{tag}.png", annotated)

        detections = []
        if r.boxes is not None:
            for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                detections.append({"class": model.names[int(cls_id)], "confidence": round(conf, 3)})
        summary[tag] = detections
        print(tag, "->", [d["class"] for d in detections])

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
