"""
Depth-aware surround view via monocular depth estimation (MiDaS-small) +
classical inverse fisheye projection, splatted into a shared top-down
canvas with a per-cell z-buffer (nearest-wins) instead of alpha-feather
blending.

Why this should help where the flat-ground pipeline can't:
- The flat-ground pipeline assumes every pixel shows a Z=0 ground point.
  Anything with real height gets dragged outward along the camera ray --
  the well-documented smearing artifact.
- Here, each pixel's ESTIMATED real-world distance from the camera (via
  monocular depth) replaces the Z=0 assumption, so a car roof or street
  sign gets placed near its own true (X, Y) footprint instead of smeared
  toward the canvas edge.
- Overlapping cameras seeing the same real point should (if depth is
  decent) place it at nearly the same canvas cell; whichever camera's
  point is physically CLOSER to it wins (z-buffer). That resolves overlap
  by actual proximity instead of a blind blend -- no ghosting by
  construction, to the extent the depth estimate is accurate.

Caveats, stated plainly because they matter and bound what to expect:
- MiDaS was trained on ordinary rectilinear photos, not fisheye images.
  Feeding it a fisheye frame directly is out-of-distribution, especially
  near the image edges where distortion is most extreme -- expect the
  depth estimate to degrade exactly where the fisheye distorts most.
- MiDaS's raw output is RELATIVE (inverse) depth with an unknown/arbitrary
  scale and offset, not metric meters. This calibrates it per-image using
  the one region already trusted completely: flat ground near the car
  (same region the "bowl" experiment used), fitting a simple affine
  correction. Everywhere else is only as good as that fit generalizes --
  that's a plausibility check, not a guarantee.
"""

import sys

import cv2
import numpy as np
import torch

from paths import WOODSCAPE_PROJECTION_DIR
sys.path.insert(0, WOODSCAPE_PROJECTION_DIR)

_MODEL = None
_TRANSFORM = None


def load_midas():
    global _MODEL, _TRANSFORM
    if _MODEL is None:
        _MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _MODEL.eval()
        _TRANSFORM = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    return _MODEL, _TRANSFORM


def estimate_relative_depth(image_bgr):
    """Returns a disparity-like map (bigger = closer), same H x W as input."""
    model, transform = load_midas()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image_bgr.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def calibrate_disparity_to_metric(disparity, cam, extent_m, flat_radius_m=3.0, n_samples=2000):
    """
    Fits norm ~ a * (1/disparity) + b using points near enough to THIS
    camera (within flat_radius_m of the camera's own mount position, not
    the vehicle origin -- a forward-mounted camera can't see the ground
    right at the rear axle, it's behind it) that the flat-ground
    assumption is trusted. "norm" (metric distance from the camera) is
    then known exactly. Returns (a, b) or None if too few valid trusted
    samples land inside the image.
    """
    rng = np.random.default_rng(0)
    xs = rng.uniform(-extent_m, extent_m, n_samples)
    ys = rng.uniform(-extent_m, extent_m, n_samples)
    cam_x, cam_y = cam.translation[0], cam.translation[1]
    mask = (xs - cam_x) ** 2 + (ys - cam_y) ** 2 <= flat_radius_m ** 2
    xs, ys = xs[mask], ys[mask]

    world_points = np.stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)], axis=1)
    pixels = cam.project_3d_to_2d(world_points)
    u, v = pixels[:, 0], pixels[:, 1]
    img_h, img_w = disparity.shape
    valid = ~np.isnan(u) & ~np.isnan(v) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if valid.sum() < 20:
        return None

    u_valid, v_valid = u[valid], v[valid]
    cam_pos = cam.translation
    ground_points = np.stack([xs[valid], ys[valid], np.zeros_like(xs[valid])], axis=1)
    true_norm = np.linalg.norm(ground_points - cam_pos, axis=1)

    disp_at_points = disparity[v_valid.astype(int), u_valid.astype(int)]
    disp_at_points = np.clip(disp_at_points, 1e-3, None)
    inv_disp = 1.0 / disp_at_points

    A = np.stack([inv_disp, np.ones_like(inv_disp)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, true_norm, rcond=None)
    return a, b


def ground_to_canvas(X, Y, extent_m, resolution_px):
    """Forward direction of build_ground_grid's linspace mapping."""
    row = (extent_m - X) / (2 * extent_m) * (resolution_px - 1)
    col = (extent_m - Y) / (2 * extent_m) * (resolution_px - 1)
    return row, col


def splat_camera_into_canvas(image, cam, extent_m, resolution_px, canvas_color, canvas_dist,
                              flat_radius_m=3.0, pixel_stride=2, max_height_m=4.0):
    """
    Estimates depth for this camera's raw image, converts every (strided)
    pixel to a world (X, Y, Z) point, and writes its color into
    canvas_color at the corresponding cell IF it's closer to the vehicle
    than whatever is already there (canvas_dist) -- the z-buffer step.
    Mutates canvas_color / canvas_dist in place.
    """
    disparity = estimate_relative_depth(image)
    calib = calibrate_disparity_to_metric(disparity, cam, extent_m, flat_radius_m)
    if calib is None:
        return
    a, b = calib

    h, w = disparity.shape
    vs, us = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
    us_flat, vs_flat = us.ravel(), vs.ravel()

    disp_vals = np.clip(disparity[vs_flat, us_flat], 1e-3, None)
    norms = a / disp_vals + b
    norms = np.clip(norms, 0.1, extent_m * 3)  # discard absurd extrapolations (e.g. sky pixels)

    screen_points = np.stack([us_flat, vs_flat], axis=1).astype(np.float64)
    world_points = cam.project_2d_to_3d(screen_points, norms)
    X, Y, Z = world_points[:, 0], world_points[:, 1], world_points[:, 2]

    dist_from_vehicle_origin = np.sqrt(X ** 2 + Y ** 2)
    in_bounds = (
        (np.abs(X) <= extent_m) & (np.abs(Y) <= extent_m)
        & (Z >= -0.5) & (Z <= max_height_m)
    )
    if not in_bounds.any():
        return

    rows, cols = ground_to_canvas(X[in_bounds], Y[in_bounds], extent_m, resolution_px)
    rows = np.clip(rows.astype(int), 0, resolution_px - 1)
    cols = np.clip(cols.astype(int), 0, resolution_px - 1)
    dists = dist_from_vehicle_origin[in_bounds]
    colors = image[vs_flat[in_bounds], us_flat[in_bounds]]

    cell_index = rows * resolution_px + cols
    flat_dist_buffer = canvas_dist.reshape(-1)
    flat_color_buffer = canvas_color.reshape(-1, 3)

    # z-buffer: this point only wins its cell if it's closer than whatever
    # is already recorded there (from this camera or an earlier one).
    current_best = flat_dist_buffer[cell_index]
    wins = dists < current_best
    flat_dist_buffer[cell_index[wins]] = dists[wins]
    flat_color_buffer[cell_index[wins]] = colors[wins]


def build_depth_aware_surround_view(frame_ids, img_dir, calib_dir, extent_m=6.0, resolution_px=1000,
                                     flat_radius_m=3.0):
    from projection import read_cam_from_json

    canvas_color = np.zeros((resolution_px, resolution_px, 3), dtype=np.uint8)
    canvas_dist = np.full((resolution_px, resolution_px), np.inf, dtype=np.float64)

    for cam_name in ["FV", "RV", "MVL", "MVR"]:
        idx = frame_ids[cam_name]
        image = cv2.imread(f"{img_dir}/{idx}_{cam_name}.png")
        cam = read_cam_from_json(f"{calib_dir}/{idx}_{cam_name}.json")
        splat_camera_into_canvas(image, cam, extent_m, resolution_px, canvas_color, canvas_dist, flat_radius_m)

    return canvas_color
