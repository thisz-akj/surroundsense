"""
Same comparison as experiment_feature_align.py, but testing the corrected
approach: dense local template matching + ground-plane similarity-transform
correction applied to the query grid (ground_plane_align.py), instead of
warping the rendered BEV image with a homography (which failed).
"""

import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, EXPERIMENTS_DIR, SAMPLE_10_PATH  # noqa: E402
from woodscape_surround_view import stitch_surround_view  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "sample10")
SAMPLES_PATH = SAMPLE_10_PATH


def main():
    os.makedirs(f"{OUT_DIR}/ground_plane_aligned", exist_ok=True)

    samples = json.load(open(SAMPLES_PATH))
    cams = ["FV", "RV", "MVL", "MVR"]

    all_before, all_after = [], []

    for s in samples:
        frame_ids = {c: s[c] for c in cams}
        tag = f"idx{s['index']:04d}"

        aligned_img, debug = stitch_surround_view(
            frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000,
            align_features="ground_plane", return_debug=True,
        )
        cv2.imwrite(f"{OUT_DIR}/ground_plane_aligned/{tag}.png", aligned_img)

        before = debug["seam_score_before"]
        after = debug["seam_score_after"]
        fits = debug["ground_plane_fits"]
        print(f"{tag}: fits={fits}")
        print(f"        before={before}")
        print(f"        after ={after}")
        all_before.append(before)
        all_after.append(after)

    print("\n=== Average seam-disagreement score per pair (lower = better) ===")
    pairs = all_before[0].keys()
    for pair in pairs:
        b_vals = [b[pair] for b in all_before if b[pair] is not None]
        a_vals = [a[pair] for a in all_after if a[pair] is not None]
        if not b_vals or not a_vals:
            print(f"{pair}: insufficient overlap in sample set")
            continue
        b_avg = sum(b_vals) / len(b_vals)
        a_avg = sum(a_vals) / len(a_vals)
        pct = 100 * (b_avg - a_avg) / b_avg if b_avg else 0
        print(f"{pair}: before={b_avg:.2f}  after={a_avg:.2f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
