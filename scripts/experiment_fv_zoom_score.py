"""
Quantitative companion to experiment_fv_zoom.py: for each candidate FV
intrinsic-scale factor, measures the FV-MVL and FV-MVR seam disagreement
score (mean abs pixel diff over the region both cameras see) across all 20
sandbox frames, not just the single idx0108 crop -- a scalar-scale
correction should help (or hurt) consistently across frames if the "front
view looks small" theory is a real, fixable calibration-scale effect rather
than a one-frame illusion.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, SAMPLE_20_PATH  # noqa: E402
from woodscape_surround_view import (  # noqa: E402
    build_bowl_grid, camera_to_bev, compute_gain_correction, read_cam_from_json,
)
from feature_align import seam_disagreement_score  # noqa: E402

SAMPLES_PATH = SAMPLE_20_PATH
SCALES = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 1.00]
EXTENT_M = 6.0
RES_PX = 1000


def render_patches(frame_ids, grid_x, grid_y, grid_z, fv_scale):
    patches, alphas = {}, {}
    for cam_name in ["FV", "RV", "MVL", "MVR"]:
        idx = frame_ids[cam_name]
        image = cv2.imread(f"{IMG_DIR}/{idx}_{cam_name}.png")
        cam = read_cam_from_json(f"{CALIB_DIR}/{idx}_{cam_name}.json")
        if cam_name == "FV":
            cam.lens.coefficients = cam.lens.coefficients * fv_scale
        patch = camera_to_bev(image, cam, grid_x, grid_y, grid_z, cam_name=cam_name)
        patches[cam_name] = patch[:, :, :3]
        alphas[cam_name] = patch[:, :, 3].astype(np.float32) / 255.0
    gains = compute_gain_correction(patches, alphas)
    for c in patches:
        patches[c] = np.clip(patches[c].astype(np.float32) * gains[c], 0, 255)
    return patches, alphas


def main():
    samples = json.load(open(SAMPLES_PATH))
    grid_x, grid_y, grid_z = build_bowl_grid(EXTENT_M, RES_PX, flat_radius_m=3.0, rim_height_m=2.5)

    totals = {s: {"FV-MVL": [], "FV-MVR": []} for s in SCALES}
    for s in samples:
        frame_ids = {c: s[c] for c in ["FV", "RV", "MVL", "MVR"]}

        # Fixed reference region: the FV/MVL and FV/MVR overlap masks at the
        # UNCHANGED (scale=1.0) calibration. Every scale is then scored only
        # over these same pixels, so a growing/shrinking mask can't be
        # mistaken for a real alignment change.
        base_patches, base_alphas = render_patches(frame_ids, grid_x, grid_y, grid_z, 1.0)
        base_overlap = {
            pair: (base_alphas[pair.split("-")[0]] > 0.5) & (base_alphas[pair.split("-")[1]] > 0.5)
            for pair in ["FV-MVL", "FV-MVR"]
        }

        for scale in SCALES:
            patches, alphas = render_patches(frame_ids, grid_x, grid_y, grid_z, scale)
            for pair in ["FV-MVL", "FV-MVR"]:
                a, b = pair.split("-")
                region = base_overlap[pair] & (alphas[a] > 0.5) & (alphas[b] > 0.5)
                if region.sum() < 50:
                    continue
                diff = np.abs(patches[a][region] - patches[b][region])
                totals[scale][pair].append(float(diff.mean()))
        print(f"done idx{s['index']:04d}")

    print()
    print(f"{'scale':>6} | {'FV-MVL mean':>12} | {'FV-MVR mean':>12} | {'combined':>10}")
    for scale in SCALES:
        mvl = np.mean(totals[scale]["FV-MVL"]) if totals[scale]["FV-MVL"] else float("nan")
        mvr = np.mean(totals[scale]["FV-MVR"]) if totals[scale]["FV-MVR"] else float("nan")
        combined = np.mean([mvl, mvr])
        print(f"{scale:>6.2f} | {mvl:>12.3f} | {mvr:>12.3f} | {combined:>10.3f}")


if __name__ == "__main__":
    main()
