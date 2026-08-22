"""
Blind-spot warning demo: for each of the 50 sample frames, run per-camera
detection + ground-fusion (experiment_fuse_detections.collect_detections),
draw the blind-spot zones, highlight any detection sitting inside one, and
log a warning. This is the same detect-then-fuse pipeline from before --
blind_spot_monitor.py just adds the "is this position somewhere we care
about" check on top of positions we already had.
"""

import json
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import EXPERIMENTS_DIR  # noqa: E402
from experiment_fuse_detections import (  # noqa: E402
    CANVAS_DIR, MODEL_PATH, SAMPLES_PATH, CAM_COLORS,
    MARKER_RADIUS_PX, DEFAULT_MARKER_RADIUS_PX, EXTENT_M, RES_PX,
    collect_detections, draw_footprint, ground_to_canvas,
)
from blind_spot_monitor import check_zones, draw_zones, draw_warning, draw_banner  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "blind_spot_demo_50")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    samples = json.load(open(SAMPLES_PATH))

    warnings_log = {}
    n_frames_with_warning = 0

    for s in samples:
        tag = f"idx{s['index']:04d}"
        canvas = cv2.imread(f"{CANVAS_DIR}/{tag}.png")
        if canvas is None:
            print("missing canvas for", tag)
            continue

        detections = collect_detections(model, s)
        draw_zones(canvas, ground_to_canvas)

        hits = check_zones(detections)

        for det in detections:
            radius_px = MARKER_RADIUS_PX.get(det["class"], DEFAULT_MARKER_RADIUS_PX)
            color = CAM_COLORS[det["cam"]]
            label = f"{det['class']} {det['conf']:.2f}"
            draw_footprint(canvas, det["row"], det["col"], radius_px, color, label)

        for det, zone_name in hits:
            draw_warning(canvas, det, zone_name)

        draw_banner(canvas, hits)

        cv2.imwrite(f"{OUT_DIR}/{tag}.png", canvas)

        if hits:
            n_frames_with_warning += 1
            warnings_log[tag] = [
                {"class": d["class"], "conf": round(d["conf"], 3), "zone": z,
                 "x": round(d["x"], 2), "y": round(d["y"], 2), "cam": d["cam"]}
                for d, z in hits
            ]
            print(tag, "-> WARNING:", warnings_log[tag])
        else:
            print(tag, "-> clear")

    with open(f"{OUT_DIR}/warnings.json", "w") as f:
        json.dump(warnings_log, f, indent=2)

    print(f"\n{n_frames_with_warning}/{len(samples)} frames triggered a blind-spot warning")


if __name__ == "__main__":
    main()
