"""
Feature-matching-based seam alignment for the surround-view stitcher.

Problem this targets (and ONLY this problem): ghosting/doubled edges right
at the seam between two adjacent cameras. That happens because WoodScape's
stock extrinsics (each camera's believed position/rotation) are slightly
off, so the same real-world ground point gets projected to slightly
different canvas pixels by two different cameras.

This does NOT fix (nothing feature-matching-based can):
- radial smearing of anything not on the ground (cars, railings, buildings)
- the resolution collapse far from a low-mounted camera
Both of those are separate, already-documented limitations.

Method (the classical panorama-stitching approach: detect -> match ->
estimate transform -> warp), applied locally to each overlap region:
1. FV and RV are treated as "anchors" (kept as-is).
2. For each anchor/mirror-camera pair that physically overlaps
   (FV-MVL, FV-MVR, RV-MVL, RV-MVR), crop just the shared valid region,
   detect ORB keypoints in both crops, match them, and estimate the 2D
   homography that would warp the mirror camera's pixels onto the
   anchor's pixels.
3. Average the (up to 2) homographies estimated for each mirror camera
   from its two anchor overlaps, and warp that camera's whole BEV patch
   by it before blending.

This is a real but bounded fix: it corrects the *pixel-space* symptom of
the calibration error within the overlap, not the underlying 3D extrinsics.
"""

import cv2
import numpy as np

ANCHORS = ["FV", "RV"]
MOVABLE = ["MVL", "MVR"]
OVERLAP_PAIRS = [("FV", "MVL"), ("FV", "MVR"), ("RV", "MVL"), ("RV", "MVR")]

MIN_MATCH_COUNT = 8


def _overlap_bbox(alpha_a, alpha_b, pad=20):
    mask = (alpha_a > 0) & (alpha_b > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    return x0, x1, y0, y1


def estimate_homography(patch_anchor, alpha_anchor, patch_move, alpha_move):
    """Returns a 3x3 homography mapping patch_move's pixels onto
    patch_anchor's pixel coordinates, or None if not enough matches."""
    bbox = _overlap_bbox(alpha_anchor, alpha_move)
    if bbox is None:
        return None, 0
    x0, x1, y0, y1 = bbox

    crop_anchor = cv2.cvtColor(patch_anchor[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop_move = cv2.cvtColor(patch_move[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp_a, des_a = orb.detectAndCompute(crop_anchor, None)
    kp_m, des_m = orb.detectAndCompute(crop_move, None)
    if des_a is None or des_m is None or len(kp_a) < MIN_MATCH_COUNT or len(kp_m) < MIN_MATCH_COUNT:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_m, des_a, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < MIN_MATCH_COUNT:
        return None, len(good)

    pts_move = np.float32([kp_m[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_anchor = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # matches were found in the crop's local coordinates -- shift back to
    # full-canvas coordinates before fitting the homography, so it's valid
    # to apply to the whole patch, not just the crop.
    pts_move[:, 0, 0] += x0
    pts_move[:, 0, 1] += y0
    pts_anchor[:, 0, 0] += x0
    pts_anchor[:, 0, 1] += y0

    H, mask = cv2.findHomography(pts_move, pts_anchor, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def align_movable_cameras(patches, alphas, resolution_px, verbose=False):
    """
    patches, alphas: dicts keyed by camera name (as produced by
    camera_to_bev), for exactly FV, RV, MVL, MVR.
    Returns new (patches, alphas) dicts where MVL/MVR have been warped to
    better align with FV/RV at the seams. FV/RV are returned unchanged.
    """
    out_patches = dict(patches)
    out_alphas = dict(alphas)
    report = {}

    for move_cam in MOVABLE:
        homographies, weights = [], []
        for anchor_cam in ANCHORS:
            H, inliers = estimate_homography(
                patches[anchor_cam], alphas[anchor_cam],
                patches[move_cam], alphas[move_cam],
            )
            report[f"{anchor_cam}-{move_cam}"] = inliers
            if H is not None:
                homographies.append(H)
                weights.append(inliers)

        if not homographies:
            continue  # not enough features anywhere; leave this camera untouched

        # Weighted average in log space would be more "correct" for
        # homographies, but a plain weighted mean is a reasonable, simple
        # approximation when the two estimated H's are already close to
        # identity (small calibration corrections, not large warps).
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        H_avg = sum(w * H for w, H in zip(weights, homographies))

        warped_patch = cv2.warpPerspective(
            out_patches[move_cam], H_avg, (resolution_px, resolution_px),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        warped_alpha = cv2.warpPerspective(
            out_alphas[move_cam], H_avg, (resolution_px, resolution_px),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        out_patches[move_cam] = warped_patch
        out_alphas[move_cam] = warped_alpha

    if verbose:
        print("Feature-match inlier counts per overlap pair:", report)

    return out_patches, out_alphas


def seam_disagreement_score(patches, alphas):
    """
    Quantifies ghosting: for each overlapping pair, the mean absolute pixel
    difference between the two cameras' own view of the shared ground
    region. Lower is better -- 0 would mean the two cameras render the
    overlap identically (perfect alignment + identical exposure).
    """
    scores = {}
    for cam_a, cam_b in OVERLAP_PAIRS:
        overlap = (alphas[cam_a] > 0.5) & (alphas[cam_b] > 0.5)
        if overlap.sum() < 50:
            scores[f"{cam_a}-{cam_b}"] = None
            continue
        diff = np.abs(
            patches[cam_a][overlap].astype(np.float32) - patches[cam_b][overlap].astype(np.float32)
        )
        scores[f"{cam_a}-{cam_b}"] = float(diff.mean())
    return scores
