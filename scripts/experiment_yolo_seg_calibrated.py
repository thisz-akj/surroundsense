"""
Stage 2 of the 3-step segmentation performance experiment.

Stage 1 (raw fisheye, no calibration warp at all): experiment_yolo_seg_raw.py
Stage 2 (calibrated per-camera BEV patch, BEFORE blending -- this script)
Stage 3 (final blended/stitched surround view): experiment_yolo_seg.py

This stage runs YOLO-seg on each camera's own ground-projected BEV patch:
calibration is fully applied (every pixel comes from asking "which fisheye
pixel shows this ground point," exactly like the real pipeline), but the 4
cameras are NOT yet blended into one canvas -- no gain correction, no
seam feathering, no denoising, no car silhouette. Isolates whether
detection quality drops because of the per-camera ground-plane warp
itself, or because of the blending/stitching step on top of it (stage 3).
"""

import json
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, EXPERIMENTS_DIR, YOLO_SEG_MODEL_PATH, SAMPLE_50_PATH  # noqa: E402
from woodscape_surround_view import build_bowl_grid, camera_to_bev, read_cam_from_json  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "yolo_seg_calibrated_50")
CAMS = ["FV", "RV", "MVL", "MVR"]
EXTENT_M = 6.0
RES_PX = 1000
FLAT_RADIUS_M = 3.0
RIM_HEIGHT_M = 2.5
FV_INTRINSIC_SCALE = 0.79  # keep in sync with woodscape_surround_view.py's default


def load_cam(idx, cam_name):
    cam = read_cam_from_json(f"{CALIB_DIR}/{idx}_{cam_name}.json")
    if cam_name == "FV":
        cam.lens.coefficients = cam.lens.coefficients * FV_INTRINSIC_SCALE
    return cam


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(YOLO_SEG_MODEL_PATH)
    samples = json.load(open(SAMPLE_50_PATH))
    grid_x, grid_y, grid_z = build_bowl_grid(EXTENT_M, RES_PX, FLAT_RADIUS_M, RIM_HEIGHT_M)

    summary = {}
    for s in samples:
        tag = f"idx{s['index']:04d}"
        summary[tag] = {}
        for cam_name in CAMS:
            idx = s[cam_name]
            image = cv2.imread(f"{IMG_DIR}/{idx}_{cam_name}.png")
            cam = load_cam(idx, cam_name)
            patch = camera_to_bev(image, cam, grid_x, grid_y, grid_z, cam_name=cam_name)
            bgr = patch[:, :, :3]  # invalid regions are already black (remap border = 0)

            results = model(bgr, verbose=False)
            r = results[0]
            cv2.imwrite(f"{OUT_DIR}/{tag}_{cam_name}.png", r.plot())

            detections = []
            if r.boxes is not None:
                for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    detections.append({"class": model.names[int(cls_id)], "confidence": round(conf, 3)})
            summary[tag][cam_name] = detections
        print(tag, "->", {c: [d["class"] for d in summary[tag][c]] for c in CAMS})

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
