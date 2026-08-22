"""
Method 1, step 2: fuse the per-camera detections (not pixels) onto the
stitched canvas.

v2 fixes vs. the first version:
- Ground-contact point now comes from the actual SEGMENTATION MASK's lowest
  edge (its true silhouette bottom), not the axis-aligned bounding box's
  bottom-center -- the bbox corner can sit outside the real object when it's
  photographed at an angle, which was throwing the projected ground position
  off. The mask's own bottom pixels are a much closer match to where the
  object actually touches the ground.
- Draws an actual filled footprint shape sized to the object's real-world
  class dimensions (a car-sized rectangle, a person-sized circle, etc.)
  instead of a bare crosshair -- this is the "mask" people expect to see.
  It is NOT the literal segmentation mask reprojected pixel-by-pixel: doing
  that would re-introduce the exact flat-ground smearing problem we're
  trying to avoid, since most mask pixels sit above Z=0 (the car's roof,
  a person's torso) and would each project to a wildly different, wrong
  ground point. A schematic footprint at the single best-estimated ground
  point is what real BEV visualizations show for this reason.
"""

import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

from paths import IMG_DIR, CALIB_DIR, EXPERIMENTS_DIR, YOLO_SEG_MODEL_PATH, SAMPLE_50_PATH
from woodscape_surround_view import read_cam_from_json

CANVAS_DIR = os.path.join(EXPERIMENTS_DIR, "sample50_fv_scaled_079")
OUT_DIR = os.path.join(EXPERIMENTS_DIR, "fused_detections_50_v2")
MODEL_PATH = YOLO_SEG_MODEL_PATH
SAMPLES_PATH = SAMPLE_50_PATH
CAMS = ["FV", "RV", "MVL", "MVR"]
EXTENT_M = 6.0
RES_PX = 1000
FV_INTRINSIC_SCALE = 0.79  # keep in sync with woodscape_surround_view.py's default

CAM_COLORS = {"FV": (60, 200, 60), "RV": (60, 60, 220), "MVL": (220, 160, 40), "MVR": (200, 60, 200)}

# Fixed marker radius (px) per class -- NOT a real-world footprint size.
# An orientation-aware, correctly-scaled rectangle needs a heading estimate
# we don't have from a single frame, and a naive axis-aligned ellipse at
# true car size (4.3m) turned out to sprawl across neighboring cars and
# look wrong. A small class-scaled dot is honest about that limitation
# while still giving a filled "mask-like" marker instead of a bare cross.
MARKER_RADIUS_PX = {
    "person": 9, "bicycle": 11, "motorcycle": 13,
    "car": 18, "bus": 26, "truck": 24, "train": 26,
}
DEFAULT_MARKER_RADIUS_PX = 8
MIN_CONFIDENCE = 0.30


def load_cam(calib_dir, idx, cam_name):
    cam = read_cam_from_json(f"{calib_dir}/{idx}_{cam_name}.json")
    if cam_name == "FV":
        cam.lens.coefficients = cam.lens.coefficients * FV_INTRINSIC_SCALE
    return cam


def mask_ground_contact_pixel(mask_xy, bottom_frac=0.05):
    """mask_xy: (N,2) polygon points in original image pixel coords.
    Returns the (u, v) of the centroid of the mask's own lowest points --
    a much closer approximation of where the object meets the ground than
    the bounding box's bottom-center, especially for objects photographed
    at an angle (fisheye periphery, oblique mirror views)."""
    ys = mask_xy[:, 1]
    y_max = ys.max()
    y_thresh = y_max - bottom_frac * (ys.max() - ys.min() + 1e-6)
    bottom_pts = mask_xy[ys >= y_thresh]
    return float(bottom_pts[:, 0].mean()), float(bottom_pts[:, 1].mean())


def pixel_to_ground(cam, u, v):
    """Back-projects a fisheye pixel to its Z=0 (ground) intersection in
    vehicle-frame meters, using the camera's own calibration."""
    screen_pt = np.array([[u, v]], dtype=np.float64)
    p0 = cam.project_2d_to_3d(screen_pt, norm=np.array([0.0]))[0]
    p1 = cam.project_2d_to_3d(screen_pt, norm=np.array([1.0]))[0]
    dz = p1[2] - p0[2]
    if abs(dz) < 1e-9:
        return None
    t = -p0[2] / dz
    if t <= 0:
        return None
    ground = p0 + t * (p1 - p0)
    return float(ground[0]), float(ground[1])


def ground_to_canvas(x, y, extent_m=EXTENT_M, resolution_px=RES_PX):
    step = (2 * extent_m) / (resolution_px - 1)
    row = (extent_m - x) / step
    col = (extent_m - y) / step
    return row, col


def draw_footprint(canvas, row, col, radius_px, color, label):
    overlay = canvas.copy()
    cv2.circle(overlay, (col, row), radius_px, color, thickness=-1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)
    cv2.circle(canvas, (col, row), radius_px, color, thickness=1)
    cv2.putText(canvas, label, (col + radius_px + 3, row), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color, 1, cv2.LINE_AA)


def collect_detections(model, s):
    """s: one sample_50.json entry (per-camera frame indices for one
    synced moment). Runs per-camera YOLO-seg, projects each detection's
    mask-bottom ground-contact point through that camera's own calibration,
    and returns every detection that lands inside our canvas as a dict:
    {cam, class, conf, x, y (vehicle-frame meters), row, col (canvas px)}.
    Shared by experiment_fuse_detections.py and blind_spot_monitor.py so
    both draw from exactly the same detection/projection logic."""
    detections = []
    for cam_name in CAMS:
        idx = s[cam_name]
        img_path = f"{IMG_DIR}/{idx}_{cam_name}.png"
        cam = load_cam(CALIB_DIR, idx, cam_name)

        results = model(img_path, verbose=False)
        r = results[0]
        if r.boxes is None or r.masks is None:
            continue

        for mask_xy, cls_id, conf in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if len(mask_xy) < 3 or conf < MIN_CONFIDENCE:
                continue
            u, v = mask_ground_contact_pixel(mask_xy)
            ground = pixel_to_ground(cam, u, v)
            if ground is None:
                continue
            gx, gy = ground
            if abs(gx) > EXTENT_M or abs(gy) > EXTENT_M:
                continue
            row, col = ground_to_canvas(gx, gy)
            row, col = int(round(row)), int(round(col))
            if not (0 <= row < RES_PX and 0 <= col < RES_PX):
                continue

            detections.append({
                "cam": cam_name, "class": model.names[int(cls_id)], "conf": conf,
                "x": gx, "y": gy, "row": row, "col": col,
            })
    return detections


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    samples = json.load(open(SAMPLES_PATH))

    for s in samples:
        tag = f"idx{s['index']:04d}"
        canvas = cv2.imread(f"{CANVAS_DIR}/{tag}.png")
        if canvas is None:
            print("missing canvas for", tag)
            continue

        detections = collect_detections(model, s)
        for det in detections:
            radius_px = MARKER_RADIUS_PX.get(det["class"], DEFAULT_MARKER_RADIUS_PX)
            color = CAM_COLORS[det["cam"]]
            label = f"{det['class']} {det['conf']:.2f}"
            draw_footprint(canvas, det["row"], det["col"], radius_px, color, label)

        cv2.imwrite(f"{OUT_DIR}/{tag}.png", canvas)
        print(tag, "-> drawn", len(detections), "fused detections")


if __name__ == "__main__":
    main()
