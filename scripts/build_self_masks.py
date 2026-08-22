"""
Builds a static "ego-vehicle body" mask per camera: the part of hood,
mirror stalk, roofline etc. that's always in that camera's view because
it's mounted on the car, not on the scene.

Key insight: the vehicle body is the ONE thing that's identical in every
frame from a given camera, while everything else (road, buildings, other
cars) changes as the vehicle drives. So across many random frames, taking
the per-pixel standard deviation isolates it automatically: low std = the
same pixels every time = part of the car; high std = the changing world.

The fisheye's own black corners (rectangular sensor frame, circular actual
field of view) are also constant across frames and get caught by the same
test -- which is fine, they should be masked out too.
"""

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, SELF_MASKS_DIR  # noqa: E402

OUT_DIR = SELF_MASKS_DIR
CAMS = ["FV", "RV", "MVL", "MVR"]
N_SAMPLES = 50
STD_THRESHOLD = 10.0  # pixels with std below this, across N_SAMPLES frames, are "static"


def build_mask_for_camera(cam_name, n_samples=N_SAMPLES, seed=0):
    random.seed(seed)
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(f"_{cam_name}.png")]
    sample = random.sample(files, min(n_samples, len(files)))

    stack = None
    for i, f in enumerate(sample):
        img = cv2.imread(os.path.join(IMG_DIR, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if stack is None:
            stack = np.zeros((len(sample), *img.shape), dtype=np.float32)
        stack[i] = img

    std = stack.std(axis=0)
    static_mask = (std < STD_THRESHOLD).astype(np.uint8) * 255

    # Clean up: keep only sizeable static blobs (real vehicle-body regions
    # are large and contiguous; isolated static-looking pixels elsewhere
    # are more likely coincidence -- e.g. a uniformly gray sky patch that
    # happened to look similar across the sample).
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = 0.002 * cleaned.size  # drop specks smaller than 0.2% of the frame
    final_mask = np.zeros_like(cleaned)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == label] = 255

    return final_mask, std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cam_name in CAMS:
        mask, std = build_mask_for_camera(cam_name)
        cv2.imwrite(f"{OUT_DIR}/{cam_name}_self_mask.png", mask)
        std_vis = cv2.applyColorMap(
            np.clip(std, 0, 60).astype(np.uint8) * 4, cv2.COLORMAP_INFERNO
        )
        cv2.imwrite(f"{OUT_DIR}/{cam_name}_std_debug.png", std_vis)
        coverage = 100 * mask.sum() / (255 * mask.size)
        print(f"{cam_name}: masked {coverage:.2f}% of frame as static/self")


if __name__ == "__main__":
    main()
