"""
Tests the user's hypothesis: "if I zoom in front view... everything from
front view looks small, so scale doesn't match."

In this pipeline we never resize the raw image before projecting -- we ask
the calibration model directly "which pixel shows this ground point," so a
naive image resize wouldn't change anything. The equivalent operation in the
calibration math is scaling the radial polynomial's coefficients: rho(theta)
= k1*theta + k2*theta^2 + k3*theta^3 + k4*theta^4 maps an incoming ray angle
to a pixel radius from the principal point. Multiplying all 4 coefficients
by a constant factor s scales every rho by s -- i.e. it moves every point s
times farther from (s>1) or closer to (s<1) FV's own optical center, exactly
a digital zoom in/out, without touching RV/MVL/MVR.

Sweeps s and renders both the full canvas and a tight crop on the known
FV/MVR seam (the lane-line jump around img[60:220, 500:750]) for idx0108,
so the effect on scale-matching at the seam can be checked directly.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, EXPERIMENTS_DIR, SAMPLE_20_PATH  # noqa: E402
from woodscape_surround_view import stitch_surround_view  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "fv_zoom_sweep")
SAMPLES_PATH = SAMPLE_20_PATH
TEST_INDEX = 108
SCALES = [round(s, 4) for s in np.linspace(0.75, 0.85, 10)]
SEAM_CROP = (slice(60, 220), slice(500, 750))  # row, col


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = json.load(open(SAMPLES_PATH))
    cams = ["FV", "RV", "MVL", "MVR"]
    s = next(s for s in samples if s["index"] == TEST_INDEX)
    frame_ids = {c: s[c] for c in cams}

    for scale in SCALES:
        img = stitch_surround_view(
            frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000,
            surface="bowl", flat_radius_m=3.0, rim_height_m=2.5,
            intrinsic_scale_by_camera={"FV": scale},
        )
        tag = f"scale{scale:.2f}"
        cv2.imwrite(f"{OUT_DIR}/full_{tag}.png", img)
        crop = img[SEAM_CROP]
        cv2.imwrite(f"{OUT_DIR}/seam_{tag}.png", crop)
        print("saved", tag)


if __name__ == "__main__":
    main()
