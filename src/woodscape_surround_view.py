"""
WoodScape 4-camera surround view (bird's-eye) stitcher
=======================================================

Uses Valeo's OWN calibration/projection code (scripts/calibration/projection.py
from https://github.com/valeoai/WoodScape) rather than reimplementing the
fisheye math -- verified against their sample front.json/front.jpg.

Adapted from the original reference script for this machine's actual layout:
- images live in rgb_images/, calibration JSONs live in calibration_data/
  (two separate directories, not one shared data_dir)
- the 4 cameras' filename index counters are NOT synchronized in this
  WoodScape RGB export (confirmed empirically: FV/RV/MVL/MVR use disjoint
  index ranges). True synchronization was recovered instead via the
  `timestamp` field inside vehicle_data/rgb_images/*.json, by finding the
  4 filenames (one per camera) whose timestamps are closest together.
  So frame_id is now a per-camera dict, not a single shared string.

Concept recap
-------------
For every (X, Y) point on an imaginary flat grid on the ground around the
car (vehicle coordinates: X=forward, Y=left, Z=up, origin at the rear
axle midpoint), we ask each camera's calibration "which pixel in your
fisheye image shows this ground point?" via cam.project_3d_to_2d(). We
then sample that pixel with cv2.remap. Since all 4 cameras answer this
question using the SAME ground grid, their 4 answers are automatically
aligned -- no separate "stitching alignment" step is needed, only
blending where two cameras' valid regions overlap.
"""

import os
import sys
import numpy as np
import cv2

from paths import WOODSCAPE_PROJECTION_DIR, IMG_DIR, CALIB_DIR, OUTPUTS_DIR
sys.path.insert(0, WOODSCAPE_PROJECTION_DIR)
from projection import read_cam_from_json  # noqa: E402


def build_ground_grid(extent_m=10.0, resolution_px=1000):
    """
    Square grid centered on the vehicle's rear-axle midpoint (the origin
    of WoodScape's vehicle coordinate frame).
    Returns grid_x, grid_y (vehicle-frame meters) shaped (res, res), where
    row 0 = farthest forward (+X), so the output image reads with the car
    facing "up", like a typical top-down parking-assist view.
    """
    lin = np.linspace(extent_m, -extent_m, resolution_px)   # +X (fwd) at top
    lin_y = np.linspace(extent_m, -extent_m, resolution_px)  # +Y (left) at left
    grid_x, grid_y = np.meshgrid(lin, lin_y, indexing="ij")
    return grid_x, grid_y


def build_bowl_grid(extent_m=10.0, resolution_px=1000, flat_radius_m=3.0, rim_height_m=2.5):
    """
    Same (X, Y) layout as build_ground_grid, but the surface being sampled
    is a bowl instead of a flat plane: flat (Z=0) within flat_radius_m of
    the car, then rising as a parabola out to rim_height_m at the grid's
    outer edge. This is what most production "3D surround view" displays
    actually project onto -- not because it's geometrically correct (nothing
    is really shaped like this), but because it curves the low-confidence
    far field up and away from the primary view, and because a rising
    surface is hit by less grazing camera rays than a flat far field is
    (see the resolution-collapse measurement in the surrounding project
    notes), which in practice reduces the flat-field blockiness too.
    """
    grid_x, grid_y = build_ground_grid(extent_m, resolution_px)
    r = np.sqrt(grid_x ** 2 + grid_y ** 2)
    beyond = np.clip(r - flat_radius_m, 0, None)
    max_beyond = max(extent_m * np.sqrt(2) - flat_radius_m, 1e-6)
    grid_z = rim_height_m * (beyond / max_beyond) ** 2
    return grid_x, grid_y, grid_z


def camera_to_bev(image, cam, grid_x, grid_y, grid_z=None, cam_name=None):
    """
    Projects the ground (or bowl, if grid_z is given) grid into this
    camera's fisheye image and samples it, returning a BGRA patch (alpha=0
    wherever the ray falls outside the source image or behind the camera,
    or -- if cam_name is given -- shows the vehicle's own body; see
    self_mask_geometry.py).
    """
    h, w = grid_x.shape
    ones = np.ones_like(grid_x)
    zeros = grid_z if grid_z is not None else np.zeros_like(grid_x)

    world_points = np.stack(
        [grid_x.ravel(), grid_y.ravel(), zeros.ravel(), ones.ravel()], axis=1
    )
    pixels = cam.project_3d_to_2d(world_points)  # (N, 2), NaN = invalid
    u = pixels[:, 0].reshape(h, w).astype(np.float32)
    v = pixels[:, 1].reshape(h, w).astype(np.float32)

    img_h, img_w = image.shape[:2]
    valid = (
        ~np.isnan(u) & ~np.isnan(v)
        & (u >= 0) & (u < img_w)
        & (v >= 0) & (v < img_h)
    )
    if cam_name is not None:
        from self_mask_geometry import self_mask_valid
        valid &= self_mask_valid(u, v, cam_name)
    u_safe = np.where(valid, u, -1).astype(np.float32)
    v_safe = np.where(valid, v, -1).astype(np.float32)

    sampled = cv2.remap(
        image, u_safe, v_safe,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    alpha = (valid.astype(np.float32) * 255).astype(np.uint8)
    bgra = cv2.cvtColor(sampled, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def compute_gain_correction(patches, alphas):
    """
    Each camera auto-exposes independently, so the same physical patch of
    ground can come out noticeably brighter from one camera than another --
    producing a visible brightness "step" right at the seam even though the
    geometry lines up perfectly.

    A tempting fix is to compare brightness only within the small region two
    adjacent cameras both see (the true overlap) and solve for gains that
    match it exactly. In practice that region mixes near-field and far-field
    ground seen at very different angles (asphalt looks different close up
    vs. far away under the same lighting), so the raw overlap statistics
    from different camera pairs can disagree about which camera is "really"
    brighter -- an inconsistent system, not just a noisy one.

    Instead this uses each camera's OWN overall brightness (over its whole
    valid ground region) and nudges every camera toward the group's average
    brightness. It won't perfectly erase a seam between two cameras that
    still disagree after equalizing, but it reliably kills the common case
    of "one camera is globally washed out or too dark relative to the
    others" without the fragile per-pair overlap matching above.
    """
    cams = list(patches.keys())
    means = {}
    for c in cams:
        valid = alphas[c] > 0
        means[c] = float(patches[c][valid].mean()) if valid.any() else 0.0

    nonzero = [m for m in means.values() if m > 1e-3]
    if not nonzero:
        return {c: 1.0 for c in cams}

    target = float(np.mean(nonzero))
    gains = {}
    for c in cams:
        gains[c] = np.clip(target / means[c], 0.5, 2.0) if means[c] > 1e-3 else 1.0
    return gains


def draw_car_silhouette(result, extent_m, resolution_px,
                         car_length_m=4.3, car_width_m=1.8):
    """
    The ground directly under the car is never seen by any of the 4
    cameras (they all point outward/downward past the body, not straight
    down through it) -- that region of the canvas is genuinely unobserved,
    not a stitching failure. Drawing a car icon there is just being honest
    about that blind spot instead of leaving unexplained black pixels.
    """
    px_per_m = resolution_px / (2 * extent_m)
    cx, cy = resolution_px // 2, resolution_px // 2
    half_len = int(car_length_m * px_per_m / 2)
    half_wid = int(car_width_m * px_per_m / 2)

    overlay = result.copy()
    body_top_left = (cx - half_wid, cy - half_len)
    body_bottom_right = (cx + half_wid, cy + half_len)
    cv2.rectangle(overlay, body_top_left, body_bottom_right, (60, 60, 60), thickness=-1)

    # Small triangle at the top (which is +X / forward, per build_ground_grid)
    # so the icon's facing direction matches the canvas's own orientation.
    nose = np.array([
        [cx - half_wid, cy - half_len],
        [cx + half_wid, cy - half_len],
        [cx, cy - half_len - int(0.3 * px_per_m)],
    ])
    cv2.fillConvexPoly(overlay, nose, (60, 60, 60))
    cv2.rectangle(overlay, body_top_left, body_bottom_right, (200, 200, 200), thickness=2)

    return overlay


def stitch_surround_view(frame_ids, img_dir, calib_dir, extent_m=10.0, resolution_px=1000,
                          apply_gain_correction=True, draw_car=True, align_features=None,
                          correspondence_fn=None,
                          surface="flat", flat_radius_m=3.0, rim_height_m=2.5, denoise=True,
                          return_debug=False, intrinsic_scale_by_camera=None):
    """
    frame_ids: dict mapping camera name -> its own frame index string, e.g.
        {"FV": "05098", "RV": "05101", "MVL": "05099", "MVR": "05100"}
    (indices differ per camera because this dataset's filename counters
    are not synchronized across cameras -- see module docstring)

    align_features: None (off), "bev_homography" (retired first attempt --
    see feature_align.py for why it caused black gaps), or "ground_plane"
    (dense local template matching + a similarity-transform correction
    applied to the query grid -- see ground_plane_align.py). Neither mode
    affects off-ground-plane smearing or far-field resolution; those are
    different, unrelated limitations.

    surface: "flat" (the original Z=0 ground plane) or "bowl" (curves up
    beyond flat_radius_m to rim_height_m at the outer edge -- see
    build_bowl_grid).

    intrinsic_scale_by_camera: defaults to {"FV": 0.79} -- FV's own
    calibration scaled toward its own principal point, the user-confirmed
    best visual match after comparing a full zoom sweep (see
    experiment_fv_zoom.py and docs/surround_view_pipeline.md).
    """
    if intrinsic_scale_by_camera is None:
        intrinsic_scale_by_camera = {"FV": 0.79}
    cams_needed = ["FV", "RV", "MVL", "MVR"]
    if surface == "bowl":
        grid_x, grid_y, grid_z = build_bowl_grid(extent_m, resolution_px, flat_radius_m, rim_height_m)
    else:
        grid_x, grid_y = build_ground_grid(extent_m, resolution_px)
        grid_z = None

    images, cams, patches, alphas = {}, {}, {}, {}
    for cam_name in cams_needed:
        idx = frame_ids[cam_name]
        img_path = f"{img_dir}/{idx}_{cam_name}.png"
        json_path = f"{calib_dir}/{idx}_{cam_name}.json"

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        cam = read_cam_from_json(json_path)
        if intrinsic_scale_by_camera and cam_name in intrinsic_scale_by_camera:
            # Scales rho(theta) = k1*theta + k2*theta^2 + ... uniformly for
            # every incoming ray, which is exactly a zoom in/out around this
            # camera's own principal point in image space -- a way to test
            # whether this camera's polynomial maps ground distance to pixel
            # radius on a different scale than its neighbors do.
            cam.lens.coefficients = cam.lens.coefficients * intrinsic_scale_by_camera[cam_name]
        images[cam_name] = image
        cams[cam_name] = cam

        patch = camera_to_bev(image, cam, grid_x, grid_y, grid_z, cam_name=cam_name)
        patches[cam_name] = patch[:, :, :3]
        alphas[cam_name] = patch[:, :, 3].astype(np.float32) / 255.0

    debug = {}
    if align_features == "bev_homography":
        from feature_align import align_movable_cameras, seam_disagreement_score
        debug["seam_score_before"] = seam_disagreement_score(patches, alphas)
        patches, alphas = align_movable_cameras(patches, alphas, resolution_px)
        debug["seam_score_after"] = seam_disagreement_score(patches, alphas)
    elif align_features == "ground_plane":
        from feature_align import seam_disagreement_score
        from ground_plane_align import (
            find_local_correspondences, fit_ground_plane_correction, apply_correction_to_grid,
        )
        find_correspondences = correspondence_fn or find_local_correspondences
        debug["seam_score_before"] = seam_disagreement_score(patches, alphas)
        debug["ground_plane_fits"] = {}
        for move_cam in ["MVL", "MVR"]:
            pooled = []
            for anchor_cam in ["FV", "RV"]:
                pooled.extend(find_correspondences(
                    patches[anchor_cam], alphas[anchor_cam], patches[move_cam], alphas[move_cam],
                ))
            transform, n_inliers = fit_ground_plane_correction(pooled, extent_m, resolution_px)
            debug["ground_plane_fits"][move_cam] = {"n_correspondences": len(pooled), "n_inliers": n_inliers}
            if transform is None:
                continue
            corrected_grid_x, corrected_grid_y = apply_correction_to_grid(grid_x, grid_y, transform)
            patch = camera_to_bev(images[move_cam], cams[move_cam], corrected_grid_x, corrected_grid_y, grid_z, cam_name=move_cam)
            patches[move_cam] = patch[:, :, :3]
            alphas[move_cam] = patch[:, :, 3].astype(np.float32) / 255.0
        debug["seam_score_after"] = seam_disagreement_score(patches, alphas)

    result = blend_patches(patches, alphas, extent_m, resolution_px,
                            apply_gain_correction=apply_gain_correction,
                            draw_car=draw_car, denoise=denoise)

    if return_debug:
        return result, debug
    return result


def blend_patches(patches, alphas, extent_m, resolution_px,
                   apply_gain_correction=True, draw_car=True, denoise=True):
    """
    The actual gain-correction + seam-blend + denoise + car-silhouette
    stage, factored out of stitch_surround_view() so it has exactly one
    implementation shared by both the file-based batch pipeline above and
    the live GStreamer pipeline in realtime/ -- the only difference
    between them is where `patches`/`alphas` (per-camera BEV patches from
    camera_to_bev(), keyed by camera name) come from.
    """
    cams_needed = list(patches.keys())
    gains = (compute_gain_correction(patches, alphas) if apply_gain_correction
             else {c: 1.0 for c in cams_needed})

    canvas = np.zeros((resolution_px, resolution_px, 3), dtype=np.float32)
    weight_sum = np.zeros((resolution_px, resolution_px), dtype=np.float32)

    POWER_BY_CAMERA = {"FV": 8, "RV": 8, "MVL": 8, "MVR": 8}

    for cam_name in cams_needed:
        corrected = np.clip(patches[cam_name].astype(np.float32) * gains[cam_name], 0, 255)

        # Weight each camera by how deep this pixel is inside ITS OWN valid
        # region (distance to the nearest invalid pixel), raised to a power
        # to sharpen dominance. A plain Gaussian-blurred edge (tried first)
        # only fades near each camera's own boundary; deep inside a big
        # shared overlap both cameras are already fully "confident" (blur
        # saturates to 1 well before reaching the middle of a wide overlap),
        # so the whole shared region got a flat 50/50 blend of two different
        # (parallax-shifted) views of the same object -- a visible double
        # image. This distance-based weight instead keeps growing the
        # deeper you are into a camera's own region, so in an overlap
        # whichever camera sees that point more centrally dominates almost
        # completely, with blending only right at the crossover line.
        binary_alpha = (alphas[cam_name] > 0.5).astype(np.uint8)
        distance = cv2.distanceTransform(binary_alpha, cv2.DIST_L2, 5).astype(np.float32)
        feather = distance ** POWER_BY_CAMERA.get(cam_name, 8)

        for c in range(3):
            canvas[:, :, c] += corrected[:, :, c] * feather
        weight_sum += feather

    safe_weight = np.where(weight_sum > 1e-6, weight_sum, 1.0)
    result = (canvas / safe_weight[:, :, None]).astype(np.uint8)
    result = np.where(weight_sum[:, :, None] > 1e-6, result, 0).astype(np.uint8)

    if denoise:
        # Edge-preserving smoothing on the finished composite. Tried
        # denoising each raw fisheye image first instead, but a fixed-size
        # kernel there gets "diluted" by the near-field upsampling (the
        # same few source pixels get stretched across many canvas pixels),
        # so it barely touched the noisiest (closest-to-car) regions.
        # Denoising the final canvas at a fixed kernel size smooths
        # consistently everywhere regardless of local magnification.
        result = cv2.bilateralFilter(result, d=11, sigmaColor=100, sigmaSpace=11)

    if draw_car:
        result = draw_car_silhouette(result, extent_m, resolution_px)

    return result


if __name__ == "__main__":
    # Recovered via timestamp-matching in vehicle_data/rgb_images/*.json --
    # all 4 share the exact same "timestamp" value (999844), i.e. true
    # zero time-skew synchronization despite the mismatched filename indices.
    frame_ids = {"FV": "05098", "RV": "05101", "MVL": "05099", "MVR": "05100"}

    surround = stitch_surround_view(frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000)
    out_path = os.path.join(OUTPUTS_DIR, "surround_view_05098_05101_05099_05100.png")
    cv2.imwrite(out_path, surround)
    print("Saved surround view to", out_path)
