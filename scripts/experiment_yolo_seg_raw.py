"""
Method 1 from the ADAS discussion: run detection/segmentation on each
camera's own RAW fisheye image, BEFORE any stitching -- instead of on the
flattened, ground-plane-projected BEV canvas we tested earlier. Same
pretrained YOLO11n-seg model, same 50 sample frames, just applied per-camera
to rgb_images/{idx}_{cam}.png directly.
"""

import json
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, EXPERIMENTS_DIR, YOLO_SEG_MODEL_PATH, SAMPLE_50_PATH  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "yolo_seg_raw_50")
MODEL_PATH = YOLO_SEG_MODEL_PATH
SAMPLES_PATH = SAMPLE_50_PATH
CAMS = ["FV", "RV", "MVL", "MVR"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    samples = json.load(open(SAMPLES_PATH))

    summary = {}
    for s in samples:
        tag = f"idx{s['index']:04d}"
        summary[tag] = {}
        for cam in CAMS:
            img_path = f"{IMG_DIR}/{s[cam]}_{cam}.png"
            results = model(img_path, verbose=False)
            r = results[0]

            annotated = r.plot()
            cv2.imwrite(f"{OUT_DIR}/{tag}_{cam}.png", annotated)

            detections = []
            if r.boxes is not None:
                for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    detections.append({"class": model.names[int(cls_id)], "confidence": round(conf, 3)})
            summary[tag][cam] = detections
        print(tag, "->", {c: [d["class"] for d in summary[tag][c]] for c in CAMS})

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
