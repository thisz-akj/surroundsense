"""
Compares 5 correspondence-finding techniques for the seam-alignment
correction, all plugged into the exact same downstream pipeline
(fit a constrained similarity transform, apply it to the query grid,
bowl surface + gain correction + denoise + self-mask unchanged):

  - template_matching : dense NCC template matching (the one already in use)
  - sift               : sparse SIFT keypoints + BFMatcher + RANSAC
  - ecc                : tiled Enhanced Correlation Coefficient
  - phase_correlation  : tiled FFT phase correlation (translation only)
  - loftr              : deep, detector-free dense matching (kornia)

For each of the 10 sandbox frames and each technique, renders the final
surround view into its own folder and records the seam-disagreement
score before/after (lower = less ghosting) plus how many correspondences
were found.
"""

import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, EXPERIMENTS_DIR, SAMPLE_10_PATH  # noqa: E402
from woodscape_surround_view import stitch_surround_view  # noqa: E402
from advanced_matching import sift_correspondences, ecc_correspondences, phase_correlation_correspondences, loftr_correspondences  # noqa: E402

OUT_DIR = os.path.join(EXPERIMENTS_DIR, "matching_shootout")
SAMPLES_PATH = SAMPLE_10_PATH

TECHNIQUES = {
    "template_matching": None,
    "sift": sift_correspondences,
    "ecc": ecc_correspondences,
    "phase_correlation": phase_correlation_correspondences,
    "loftr": loftr_correspondences,
}


def main():
    samples = json.load(open(SAMPLES_PATH))
    cams = ["FV", "RV", "MVL", "MVR"]

    results = {name: {"before": [], "after": [], "n_corr": []} for name in TECHNIQUES}

    for tech_name, fn in TECHNIQUES.items():
        out_dir = f"{OUT_DIR}/{tech_name}"
        os.makedirs(out_dir, exist_ok=True)
        t_start = time.time()

        for s in samples:
            frame_ids = {c: s[c] for c in cams}
            tag = f"idx{s['index']:04d}"

            img, debug = stitch_surround_view(
                frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000,
                surface="bowl", align_features="ground_plane", correspondence_fn=fn,
                return_debug=True,
            )
            cv2.imwrite(f"{out_dir}/{tag}.png", img)

            before, after = debug["seam_score_before"], debug["seam_score_after"]
            fits = debug["ground_plane_fits"]
            results[tech_name]["before"].append(before)
            results[tech_name]["after"].append(after)
            results[tech_name]["n_corr"].append(
                {c: fits[c]["n_correspondences"] for c in fits}
            )
            print(f"[{tech_name}] {tag}: n_corr={ {c: fits[c]['n_correspondences'] for c in fits} }")

        print(f"=== {tech_name} done in {time.time()-t_start:.0f}s ===\n")

    print("\n\n========== SUMMARY (avg seam-disagreement score, lower=better) ==========")
    pairs = ["FV-MVL", "FV-MVR", "RV-MVL", "RV-MVR"]
    for tech_name in TECHNIQUES:
        before_list = results[tech_name]["before"]
        after_list = results[tech_name]["after"]
        avg_n_corr = {}
        for c in ["MVL", "MVR"]:
            vals = [n[c] for n in results[tech_name]["n_corr"] if c in n]
            avg_n_corr[c] = sum(vals) / len(vals) if vals else 0
        print(f"\n-- {tech_name} -- avg correspondences: MVL={avg_n_corr['MVL']:.1f} MVR={avg_n_corr['MVR']:.1f}")
        for pair in pairs:
            b_vals = [b[pair] for b in before_list if b.get(pair) is not None]
            a_vals = [a[pair] for a in after_list if a.get(pair) is not None]
            if not b_vals or not a_vals:
                print(f"  {pair}: insufficient overlap")
                continue
            b_avg = sum(b_vals) / len(b_vals)
            a_avg = sum(a_vals) / len(a_vals)
            pct = 100 * (b_avg - a_avg) / b_avg if b_avg else 0
            print(f"  {pair}: before={b_avg:.2f}  after={a_avg:.2f}  ({pct:+.1f}%)")

    with open(f"{OUT_DIR}/raw_results.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
