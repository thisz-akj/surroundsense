"""
Renders 20 sandbox frames with the bowl-shaped projection surface
(flat near the car, curving up to rim_height_m at the outer edge).
"""

import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, EXPERIMENTS_DIR, SAMPLE_20_PATH  # noqa: E402
from woodscape_surround_view import stitch_surround_view  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "sample20_fv_fix")
SAMPLES_PATH = SAMPLE_20_PATH


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = json.load(open(SAMPLES_PATH))
    cams = ["FV", "RV", "MVL", "MVR"]

    for s in samples:
        frame_ids = {c: s[c] for c in cams}
        tag = f"idx{s['index']:04d}"

        img = stitch_surround_view(
            frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000,
            surface="bowl", flat_radius_m=3.0, rim_height_m=2.5,
        )
        cv2.imwrite(f"{OUT_DIR}/{tag}.png", img)
        print("saved", tag)


if __name__ == "__main__":
    main()
