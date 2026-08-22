# WoodScape Surround-View Stitching — How We Got a Correct Output

This documents the working pipeline in `src/woodscape_surround_view.py`: the
core algorithm, every real problem we ran into on the way to a clean
360° stitched image, why each one happened, and exactly what fixed it (or
why it's a known, accepted limitation). Written so a future session (or a
future you) can pick this up without re-deriving any of it.

## 1. Goal

Take 4 fisheye images (front `FV`, rear `RV`, left-mirror `MVL`,
right-mirror `MVR`) from one synchronized moment and produce a single
top-down "360° parking camera" style image of the ground around the car.

## 2. The core algorithm

We do **not** undistort each fisheye image and then warp it with a
homography (the classic approach). Instead, for every camera we ask its
own calibration a direct question:

1. Build a flat grid of points on the ground, in vehicle coordinates
   (ISO 8855: X = forward, Y = left, Z = up, origin at the rear-axle
   midpoint) — `build_ground_grid()`.
2. For each camera, turn every grid point into `[X, Y, 0, 1]` and call
   `Camera.project_3d_to_2d()` (from Valeo's own `projection.py`) to get
   the fisheye pixel `(u, v)` that shows that real-world ground point.
3. `cv2.remap` samples the camera's actual image at those `(u, v)`
   coordinates. The result is one bird's-eye patch per camera, and because
   all 4 cameras answered the question against the *same* shared grid,
   their 4 answers are automatically aligned — no separate "stitching"
   step, only blending where two patches overlap.
4. Blend the 4 patches, weighting each pixel by how deep it sits inside
   that camera's own valid region (see §3.6).

This is `camera_to_bev()` + the blending loop inside
`stitch_surround_view()`.

## 3. Problems we hit, in the order we hit them, and the fix

### 3.1 Mismatched frame indices (data problem, not a code problem)

**Symptom:** an early attempt stitched `00000`, `00028`, `00044`, `00140` —
four *different* moments in time. Geometrically meaningless output.

**Root cause:** this WoodScape RGB export's per-camera filename counters
are **not synchronized** — `FV`/`RV`/`MVL`/`MVR` use disjoint index ranges.

**Fix:** match cameras by the `timestamp` field inside
`vehicle_data/rgb_images/*.json` instead of by filename index. `frame_ids`
is a per-camera dict of *different* index strings that all share one
timestamp — see `manifest.json` (542 verified synced quadruples) and
`sample_10/20/50.json` for fast-iteration subsets.

### 3.2 Resolution collapse far from the car

**Symptom:** ground far from the car rendered as a blurry, low-detail
smear.

**Root cause:** pixels-per-meter drops sharply with distance from a
low-mounted camera viewing the ground at a grazing angle. A large
`extent_m` (e.g. 10m) spreads that resolution very thin.

**Fix:** use a smaller `extent_m` (6.0m) so we're not asking cameras to
resolve ground far past where they have any real resolution left.

### 3.3 Off-ground-plane smearing — accepted, not fixed

**Symptom:** anything not sitting on the ground (parked cars, people,
buildings, curbs) stretches/smears radially, worse near the canvas edges.

**Root cause:** this is fundamental to flat-ground projection, not a bug.
We're asking "what does the ground look like here," and a camera has no
way to know a given pixel belongs to a 1.5m-tall pedestrian instead of the
road surface — it gets flattened onto Z=0 regardless, at whatever
distance the *ground* under/behind it would have been.

**Status:** documented, not fixed. Fixing this properly means abandoning
the flat-ground assumption entirely — see §5 for how real ADAS systems
avoid it.

### 3.4 Brightness step at every seam

**Symptom:** a visible brightness jump right at the border between two
cameras' patches, even in frames where the geometry lined up fine.

**Root cause:** each camera auto-exposes independently, so the same
physical patch of asphalt can come out a different brightness from two
different cameras.

**Fix tried and rejected:** matching brightness pairwise within just the
overlap region — unstable, because the overlap mixes near-field and
far-field ground under different lighting, so different camera pairs
disagreed about who's "really" brighter.

**Fix that worked:** `compute_gain_correction()` — nudge each camera's
*overall* brightness (over its whole valid region) toward the group
average. Simpler and more stable; doesn't chase the exact overlap-only
mismatch but reliably kills the common "one camera is washed out" case.

### 3.5 The car's own body/mirrors showing up as "ground"

**Fix:** `src/self_mask_geometry.py` — hardcoded per-camera geometric cutoffs
(a row cutoff for FV/RV, a diagonal line cutoff for MVL/MVR) derived by
averaging many frames to find which regions are *persistently* invalid
(the car's own body) vs. content that just varies frame to frame.

### 3.6 Seam ghosting / double-imaging in large overlaps

**Fix, part 1 — feathering → winner-take-most:** originally blended
overlaps with Gaussian-blurred masks (soft feathering). This caused visible
double-imaging in the *large* overlap regions. Switched to a **distance
transform**: each pixel's weight is `distance_to_nearest_invalid_pixel
** power`, so pixels deep inside a camera's own valid region dominate over
pixels near that camera's own edge — a "winner-take-most" blend rather
than a 50/50 average everywhere.

**Fix, part 2 — the bowl surface:** switched the sampled surface from a
flat Z=0 plane to a bowl (`build_bowl_grid()`): flat within
`flat_radius_m` (3.0m) of the car, curving up to `rim_height_m` (2.5m) at
the outer edge. This is what production "3D surround view" displays
actually use — not because it's geometrically correct, but because it
both frames the far field better and eases the grazing-angle resolution
problem from §3.2. Biggest single visual-quality win in the project.

This whole gain-correction + feather-blend + denoise + car-silhouette
stage now lives in its own function, `blend_patches()`, factored out of
`stitch_surround_view()` so the live pipeline in `realtime/` can call the
exact same blend implementation on frames from a GStreamer feed instead
of files on disk — one implementation, two frame sources. Verified
byte-identical output before/after the extraction.

### 3.7 Sensor grain

**Fix:** `cv2.bilateralFilter` on the *finished composite*, not on each
raw fisheye image first. We tried denoising the raw images, but a
fixed-size kernel there gets diluted by near-field upsampling (the same
few source pixels get stretched across many canvas pixels near the car),
so it barely touched the noisiest regions. Denoising the final canvas at
a fixed kernel size smooths consistently everywhere regardless of local
magnification.

### 3.8 The front camera (FV) seam — the hard one

**Symptom:** FV's seams against MVL/MVR looked visibly wrong — lane lines
jumping sideways at the seam, described by direct visual inspection as
"compressed and fitted, scaled down... road not matching, cars not
aligned, nothing blending." This was *not* a brightness problem (that was
already fixed by §3.4) — it was geometric.

**What we ruled out, in order, each with real testing (not guessing):**

1. **FV's own calibration having a fixed extrinsic bias.** Tested by
   comparing FV's translation/rotation against RV/MVL/MVR's — found only
   noise, no consistent offset.
2. **Timing/sync drift** (FV captured at a subtly different instant than
   its "synced" partners despite sharing a timestamp). Checked the CAN
   bus fields (`ego_speed`, `ego_steering`, etc.) in `vehicle_data/` for
   FV/MVL/MVR at the matched timestamp — found them **byte-identical**,
   ruling this out completely.
3. **A 2D ground-plane correction transform**, fit three different ways:
   - Dense NCC template matching between FV's and neighbors' rendered BEV
     patches → similarity transform (rotation + scale + translation) via
     RANSAC. Only ~27% inlier consistency, and applying it had no visible
     effect on the actual seam.
   - Re-verifying the exact returned transform matrix (ruling out a
     transcription error) — identical no-effect result.
   - A pure rotation+scale fit pivoted at FV's own camera position
     (Procrustes/Kabsch) instead of the canvas origin — converged to ~0°
     rotation with *worse* (12%) inlier consistency than attempt 1.
   - **LoFTR** (a deep, detector-free feature matcher) instead of template
     matching, pooling correspondences across 10 frames to get ~10x more
     points (2233 vs 226). Still failed to converge to anything stable —
     per-frame fits ranged from -14.6° to +17.2° rotation with no
     consistent sign. Root cause: LoFTR confidently matches non-ground
     content (cars, pedestrians, shadows), and those have scene-specific
     geometric relationships that don't transfer between different real-
     world scenes — pooling across frames mixes real signal with noise.
4. **A digital "zoom" of FV's raw image before projecting.** The user's
   own hypothesis, from the visual impression that "everything from front
   view looks small, so scale doesn't match." Implemented as scaling
   FV's intrinsic radial polynomial coefficients (`k1..k4`) by a constant
   factor — mathematically exactly a zoom in/out around FV's own
   principal point (see `scripts/experiment_fv_zoom.py`). Tested at scale 0.5
   through 1.15: **visually no effect whatsoever**, even at a 2x swing,
   and the seam-disagreement score dropped monotonically with no minimum
   (the signature of a metric artifact, not a real fix) — because our
   pipeline queries "which pixel shows this ground point," so a ground
   point's position in the *output* canvas is fixed by its real-world
   (X, Y) regardless of any per-camera scale factor. There is no free
   "camera looks smaller" knob in this design.

**What actually fixed it:** despite ruling out a *uniform* geometric
scale effect (above), the user directly compared full renders across a
fine scale sweep (0.75–0.85) and identified **FV intrinsic scale = 0.79**
as visibly the best match by eye, confirmed across a 50-frame sample.
This is now the pipeline's default (`intrinsic_scale_by_camera={"FV":
0.79}` in `stitch_surround_view()`). Note the apparent tension with
finding #4 above: our seam-disagreement *metric* couldn't detect this as
an improvement (it kept dropping with no minimum, which we flagged as an
artifact), but the *visual* result across many frames was clearly better
at this value than at 1.0. Take the metric's silence here as a reminder
that it isn't sensitive to whatever this correction is actually fixing —
not as proof the fix is wrong. The user's own side-by-side visual
comparison across the full 50-frame sample is the actual evidence this
was validated against.

## 4. Final default configuration

```python
stitch_surround_view(
    frame_ids, img_dir, calib_dir,
    extent_m=6.0, resolution_px=1000,
    surface="bowl", flat_radius_m=3.0, rim_height_m=2.5,
    # apply_gain_correction=True, denoise=True, draw_car=True — all defaults
    # intrinsic_scale_by_camera defaults to {"FV": 0.79}
)
```

See `scripts/experiment_sample50_fv_scaled.py` for the exact 50-frame batch
render this was validated against (`outputs/experiments/sample50_fv_scaled_079/`).

## 5. What's still a known limitation

- **Off-ground-plane smearing** (§3.3) is fundamental to this approach.
  Real ADAS systems don't run detection on a flattened BEV image for
  exactly this reason — see `docs/` discussion of per-camera detection
  fused via each camera's own calibration (`src/experiment_fuse_detections.py`)
  rather than pixel-level stitching.
- A handful of frames still show a residual seam artifact; if a fully
  clean seam is ever required, the recommended next step (not yet done)
  is `Click-Calib` (github.com/lwangvaleo/click_calib) — a tool built
  specifically to refine WoodScape's extrinsic calibration for this exact
  problem via manually-clicked ground-point correspondences, which is more
  reliable than any automatic feature-matching approach on largely
  featureless asphalt. Setup for this is already prepared in
  `external/click_calib/` (see `external/click_calib/source/click_pair.py`
  and `optimize_real.py`), pending the manual point-clicking step.

## 6. File map

| File | Role |
|---|---|
| `src/woodscape_surround_view.py` | Main pipeline: grid building, per-camera projection/remap, gain correction, blending, denoising, car silhouette |
| `src/self_mask_geometry.py` | Per-camera hardcoded masks for the car's own body |
| `src/paths.py` | Every repo path, computed relative to the project root (not hardcoded to one machine) |
| `data/manifest.json`, `data/sample_10/20/50.json` | Verified synced frame quadruples (full set / fast-iteration subsets) |
| `scripts/experiment_fv_zoom.py`, `scripts/experiment_fv_zoom_score.py` | The FV intrinsic-scale sweep and its (inconclusive) quantitative scoring |
| `scripts/experiment_sample50_fv_scaled.py` | Batch render of 50 frames with the final config |
| `src/feature_align.py`, `src/ground_plane_align.py`, `src/advanced_matching.py` | Retired/ruled-out automatic seam-correction attempts (kept for reference — see §3.8) |
| `external/click_calib/` | Manual point-and-click extrinsic recalibration tool, set up but not yet run |
