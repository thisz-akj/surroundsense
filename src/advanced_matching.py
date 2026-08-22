"""
Additional correspondence-finding techniques for the seam-alignment
experiment, beyond the two already tried (feature_align.py's ORB+homography,
and ground_plane_align.py's dense NCC template matching).

Every function here has the SAME output contract as
ground_plane_align.find_local_correspondences: given two BEV patches
(anchor, movable) and their alpha masks, return a list of
(x, y, dx, dy, score) in canvas PIXEL space, where (x, y) is a point in
the movable camera's own canvas and (x+dx, y+dy) is where the anchor
camera's rendering shows the same real content. That list plugs directly
into ground_plane_align.fit_ground_plane_correction() -- the fitting and
"apply to the query grid, not the output image" logic is identical no
matter which technique found the correspondences.

ECC and phase correlation don't naturally produce sparse point pairs --
they estimate one dense/global transform for the whole overlap. To fit
the same interface (and let the exact same downstream RANSAC + apply
code handle them), each samples several points across the overlap and
reports where that single transform sends them.
"""

import cv2
import numpy as np


def _overlap_bbox(alpha_a, alpha_b, pad=20):
    mask = (alpha_a > 0) & (alpha_b > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    return x0, x1, y0, y1


# ---------------------------------------------------------------- SIFT / AKAZE

def _sparse_keypoint_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move,
                                      detector, norm_type, min_matches=8, ratio=0.75):
    bbox = _overlap_bbox(alpha_anchor, alpha_move)
    if bbox is None:
        return []
    x0, x1, y0, y1 = bbox
    crop_a = cv2.cvtColor(patch_anchor[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop_m = cv2.cvtColor(patch_move[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    kp_a, des_a = detector.detectAndCompute(crop_a, None)
    kp_m, des_m = detector.detectAndCompute(crop_m, None)
    if des_a is None or des_m is None or len(kp_a) < min_matches or len(kp_m) < min_matches:
        return []

    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des_m, des_a, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return []

    out = []
    for m in good:
        xm, ym = kp_m[m.queryIdx].pt
        xa, ya = kp_a[m.trainIdx].pt
        x, y = xm + x0, ym + y0
        dx, dy = (xa + x0) - x, (ya + y0) - y
        score = 1.0 - m.distance / 256.0  # rough normalization, only used as a relative weight
        out.append((x, y, dx, dy, max(score, 0.01)))
    return out


def sift_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move):
    detector = cv2.SIFT_create(nfeatures=1500)
    return _sparse_keypoint_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move,
                                             detector, cv2.NORM_L2)


# AKAZE is not built into opencv-python-headless (needs opencv-contrib,
# skipped here to avoid swapping the installed cv2 package given limited
# disk headroom) -- dropped from the lineup, not substituted.


# ---------------------------------------------------------------- ECC / phase correlation (tiled)
#
# Both first tried on the WHOLE overlap region and failed outright -- a
# large overlap mixes real matching ground with parallax-shifted objects
# (cars, railings, signs), and averaging correlation over that whole messy
# region gives neither method enough signal to converge. Splitting into
# small tiles and only keeping tiles that succeed is the same "many small
# patches, discard the unreliable ones" philosophy the NCC technique uses,
# just with a different per-tile registration method.

def _iter_tiles(alpha_anchor, alpha_move, tile=90, stride=60, min_coverage=0.9):
    overlap = (alpha_anchor > 0.5) & (alpha_move > 0.5)
    h, w = overlap.shape
    for y0 in range(0, h - tile, stride):
        for x0 in range(0, w - tile, stride):
            block = overlap[y0:y0 + tile, x0:x0 + tile]
            if block.mean() >= min_coverage:
                yield x0, x0 + tile, y0, y0 + tile


def ecc_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move):
    out = []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
    for x0, x1, y0, y1 in _iter_tiles(alpha_anchor, alpha_move):
        gray_a = cv2.cvtColor(patch_anchor[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_m = cv2.cvtColor(patch_move[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray_a.std() < 0.02 or gray_m.std() < 0.02:
            continue  # featureless tile, ECC has nothing to lock onto
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(gray_a, gray_m, warp, cv2.MOTION_EUCLIDEAN, criteria)
        except cv2.error:
            continue
        if cc < 0.45:
            continue
        cx, cy = (x1 - x0) / 2, (y1 - y0) / 2
        ax_ = warp[0, 0] * cx + warp[0, 1] * cy + warp[0, 2]
        ay_ = warp[1, 0] * cx + warp[1, 1] * cy + warp[1, 2]
        x, y = cx + x0, cy + y0
        dx, dy = (ax_ + x0) - x, (ay_ + y0) - y
        out.append((x, y, dx, dy, float(cc)))
    return out


def phase_correlation_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move):
    out = []
    for x0, x1, y0, y1 in _iter_tiles(alpha_anchor, alpha_move):
        gray_a = cv2.cvtColor(patch_anchor[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_m = cv2.cvtColor(patch_move[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gray_a.std() < 5 or gray_m.std() < 5:
            continue
        win = cv2.createHanningWindow((gray_a.shape[1], gray_a.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(gray_m, gray_a, win)
        if response < 0.12 or abs(dx) > (x1 - x0) / 2 or abs(dy) > (y1 - y0) / 2:
            continue  # low confidence, or a shift too big to trust from one small tile
        cx, cy = (x1 - x0) / 2 + x0, (y1 - y0) / 2 + y0
        out.append((cx, cy, dx, dy, float(response)))
    return out


# ---------------------------------------------------------------- LoFTR (deep, detector-free)

_LOFTR_MODEL = None


def _load_loftr():
    global _LOFTR_MODEL
    if _LOFTR_MODEL is None:
        import torch
        import kornia.feature as KF
        _LOFTR_MODEL = KF.LoFTR(pretrained="outdoor")
        _LOFTR_MODEL.eval()
    return _LOFTR_MODEL


def loftr_correspondences(patch_anchor, alpha_anchor, patch_move, alpha_move, min_confidence=0.5):
    import torch

    bbox = _overlap_bbox(alpha_anchor, alpha_move)
    if bbox is None:
        return []
    x0, x1, y0, y1 = bbox

    gray_a = cv2.cvtColor(patch_anchor[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(patch_move[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    model = _load_loftr()
    t_a = torch.from_numpy(gray_a).float()[None, None] / 255.0
    t_m = torch.from_numpy(gray_m).float()[None, None] / 255.0

    with torch.no_grad():
        out = model({"image0": t_m, "image1": t_a})
    mkpts0 = out["keypoints0"].cpu().numpy()  # in gray_m (movable crop)
    mkpts1 = out["keypoints1"].cpu().numpy()  # in gray_a (anchor crop)
    conf = out["confidence"].cpu().numpy()

    keep = conf >= min_confidence
    if keep.sum() < 8:
        return []

    result = []
    for (xm, ym), (xa, ya), c in zip(mkpts0[keep], mkpts1[keep], conf[keep]):
        x, y = xm + x0, ym + y0
        dx, dy = (xa + x0) - x, (ya + y0) - y
        result.append((x, y, dx, dy, float(c)))
    return result


TECHNIQUES = {
    "template_matching": None,  # handled separately -- see ground_plane_align.find_local_correspondences
    "sift": sift_correspondences,
    "ecc": ecc_correspondences,
    "phase_correlation": phase_correlation_correspondences,
    "loftr": loftr_correspondences,
}
