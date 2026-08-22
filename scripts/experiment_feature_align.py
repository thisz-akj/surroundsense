"""
Renders each of the 10 sandbox frames twice -- once with the existing
pipeline (extent_m=6, gain correction, car silhouette) and once with
feature-matching seam alignment additionally turned on -- and reports the
seam-disagreement score (mean pixel difference between two cameras' views
of the ground they both claim to see) before and after, per overlap pair.

Lower seam-disagreement score = less ghosting at that seam.
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
    os.makedirs(f"{OUT_DIR}/baseline", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/feature_aligned", exist_ok=True)

    samples = json.load(open(SAMPLES_PATH))
    cams = ["FV", "RV", "MVL", "MVR"]

    all_before, all_after = [], []

    for s in samples:
        frame_ids = {c: s[c] for c in cams}
        tag = f"idx{s['index']:04d}"

        baseline_img = stitch_surround_view(
            frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000,
            align_features=False,
        )
        cv2.imwrite(f"{OUT_DIR}/baseline/{tag}.png", baseline_img)

        aligned_img, debug = stitch_surround_view(
            frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000,
            align_features=True, return_debug=True,
        )
        cv2.imwrite(f"{OUT_DIR}/feature_aligned/{tag}.png", aligned_img)

        before = debug["seam_score_before"]
        after = debug["seam_score_after"]
        print(f"{tag}: before={before}  after={after}")
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
