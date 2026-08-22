"""
Ground-plane extrinsic correction via dense local template matching.

Why the previous attempt (feature_align.py: ORB + homography, warping the
rendered BEV image) failed:
- ORB needs distinctive corners; flat asphalt gives it almost nothing to
  work with, so matches were sparse and sometimes wrong.
- A homography can add perspective distortion that a small calibration
  error would never actually produce, and warping the finished image can
  pull real content outside the canvas, leaving black holes.

This version fixes both problems:
1. Dense, not sparse: slide a small template over a grid of positions
   across the overlap and use normalized cross-correlation (NCC) to find
   its best match nearby -- skipping any template with too little texture
   (std dev too low) instead of trusting a possibly-spurious ORB match.
2. Constrained transform: fit a similarity transform (rotation + uniform
   scale + translation -- no shear, no perspective) with RANSAC, since a
   real extrinsic error is well approximated by a small rotation +
   translation, never a perspective warp.
3. Apply the correction to the INPUT ground grid (in physical meters)
   fed into the projection, not to the rendered output image. This asks
   "what pixel shows ground point (X,Y) once the calibration error is
   accounted for" -- always a valid nearby lookup in the original fisheye
   image, so it cannot create the black gaps the image-warp approach did.
"""

import cv2
import numpy as np

ANCHORS = ["FV", "RV"]
MOVABLE = ["MVL", "MVR"]


def find_local_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move,
                                grid_spacing=30, patch_half=12, search_half=20,
                                min_texture_std=6.0, min_ncc=0.6):
    """
    Slides a small template from patch_move over a grid of canvas positions;
    for each, searches a slightly larger window in patch_anchor for the best
    NCC match. Returns list of (x, y, dx, dy, score) in canvas PIXEL space,
    where (x,y) is the query position in patch_move's own canvas and
    (x+dx, y+dy) is where the matching content was found in patch_anchor's
    canvas.
    """
    gray_anchor = cv2.cvtColor(patch_anchor, cv2.COLOR_BGR2GRAY)
    gray_move = cv2.cvtColor(patch_move, cv2.COLOR_BGR2GRAY)
    overlap = (alpha_anchor > 0.5) & (alpha_move > 0.5)
    h, w = overlap.shape

    margin = patch_half + search_half
    correspondences = []
    for y in range(margin, h - margin, grid_spacing):
        for x in range(margin, w - margin, grid_spacing):
            if not overlap[y, x]:
                continue
            template = gray_move[y - patch_half:y + patch_half + 1, x - patch_half:x + patch_half + 1]
            if template.std() < min_texture_std:
                continue  # featureless asphalt -- would only produce a noisy/spurious match

            search_region = gray_anchor[
                y - margin:y + margin + 1, x - margin:x + margin + 1
            ]
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < min_ncc:
                continue

            dx = max_loc[0] - search_half
            dy = max_loc[1] - search_half
            correspondences.append((x, y, dx, dy, max_val))

    return correspondences


def canvas_to_ground(x_px, y_px, extent_m, resolution_px):
    """Inverse of build_ground_grid's linspace mapping: canvas (col=x, row=y)
    pixel -> physical (X, Y) ground-plane meters. Uses the direct linear
    formula (not array indexing) so it works for sub-pixel float
    coordinates too, not just the integer grid positions dense template
    matching happens to produce."""
    step = (2 * extent_m) / (resolution_px - 1)
    X = extent_m - y_px * step  # row maps to X
    Y = extent_m - x_px * step  # col maps to Y
    return X, Y


def fit_ground_plane_correction(correspondences, extent_m, resolution_px, min_points=8):
    """
    Converts pixel-space correspondences to physical ground-plane (X, Y)
    pairs and fits a similarity transform (rotation + uniform scale +
    translation) mapping "ground point the movable camera's calibration
    currently assigns to this canvas cell" -> "ground point that's actually
    there" (per the anchor camera's trusted rendering).

    Returns (transform_2x3, n_inliers) or (None, 0) if too few correspondences.
    """
    if len(correspondences) < min_points:
        return None, 0

    src_pts, dst_pts = [], []
    for x, y, dx, dy, score in correspondences:
        X_move, Y_move = canvas_to_ground(x, y, extent_m, resolution_px)
        X_anchor, Y_anchor = canvas_to_ground(x + dx, y + dy, extent_m, resolution_px)
        src_pts.append([X_move, Y_move])
        dst_pts.append([X_anchor, Y_anchor])

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=0.15,
    )
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return transform, n_inliers


def invert_similarity(transform_2x3):
    """2x3 similarity transform -> its inverse, also as a 2x3 matrix."""
    A = transform_2x3[:, :2]
    b = transform_2x3[:, 2]
    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b
    return np.hstack([A_inv, b_inv.reshape(2, 1)])


def apply_correction_to_grid(grid_x, grid_y, transform_2x3):
    """
    grid_x, grid_y: the shared ground grid (physical meters) as built by
    build_ground_grid. Returns a corrected (grid_x', grid_y') such that
    querying the movable camera's ORIGINAL (uncorrected) projection at the
    corrected grid gives the right pixel for the original, uncorrected
    target ground point -- see module docstring for the derivation.
    """
    inv = invert_similarity(transform_2x3)
    # transform acts on (X, Y) pairs; grid_x/grid_y are the X/Y components
    # of every canvas cell, so apply the 2x2 linear part + translation
    # elementwise via broadcasting.
    a, b, tx = inv[0]
    c, d, ty = inv[1]
    corrected_x = a * grid_x + b * grid_y + tx
    corrected_y = c * grid_x + d * grid_y + ty
    return corrected_x, corrected_y
