# WoodScape Surround-View Stitching — Project Brief

## Goal

Take 4 fisheye camera images from the WoodScape dataset (front, rear, left-mirror,
right-mirror) from the **same synchronized frame** and produce a single stitched
360° top-down (bird's-eye / surround-view) image of the area around the vehicle —
the kind of view you see in a car's "360 camera" parking display.

## Where things are

- The WoodScape dataset has already been downloaded by the user. Locate it on
  this machine (ask the user if the path isn't obvious) — expect a structure like:
  ```
  woodscape/
  ├── rgb_images/            00001_FV.png, 00001_RV.png, 00001_MVL.png, 00001_MVR.png, ...
  ├── calibration_data/      00001_FV.json, 00001_RV.json, 00001_MVL.json, 00001_MVR.json, ...
  ├── previous_images/
  ├── semantic_annotations/
  └── ...
  ```
  `[CAM]` suffix meaning: **FV** = Front View, **RV** = Rear View,
  **MVL** = Mirror View Left, **MVR** = Mirror View Right.

- Clone the official Valeo repo to get their tested projection/calibration code —
  do NOT reimplement the fisheye math from scratch, reuse theirs:
  ```
  git clone https://github.com/valeoai/WoodScape.git
  ```
  The relevant file is `WoodScape/scripts/calibration/projection.py`. It provides:
  - `read_cam_from_json(path)` → returns a `Camera` object
  - `Camera.project_3d_to_2d(world_points)` → projects vehicle-frame 3D points
    (as Nx4 homogeneous `[X, Y, Z, 1]`, ISO 8855: X=forward, Y=left, Z=up, origin
    at rear-axle midpoint) into fisheye pixel coordinates `(u, v)`, returning
    `NaN` for invalid/out-of-view points.
  - `Camera.project_2d_to_3d(...)`, `create_img_projection_maps(...)` — not
    needed for this task but available if useful.

  There is also a demo calibration + image pair at
  `WoodScape/scripts/calibration/front.json` / `front.jpg` you can use to sanity
  check the pipeline still works on this machine before touching real data.

## Critical data requirement — READ THIS FIRST

**All 4 images must come from the exact same frame index** (e.g. `00001_FV.png`,
`00001_RV.png`, `00001_MVL.png`, `00001_MVR.png` — same `00001`). Images from
different indices show the world at different moments and cannot be
geometrically stitched — the earlier attempt on this project used 4 mismatched
frames (`00000`, `00028`, `00044`, `00140`) which was a mistake; don't repeat it.
Before running anything, verify all 4 `.png` and all 4 matching `.json`
calibration files exist for the chosen index.

## Calibration file schema (confirmed from the real repo file, not guessed)

```json
{
  "extrinsic": {
    "quaternion": [x, y, z, w],
    "translation": [X, Y, Z]
  },
  "intrinsic": {
    "aspect_ratio": 1.0,
    "cx_offset": 3.942,
    "cy_offset": -3.093,
    "height": 966.0,
    "k1": 339.749, "k2": -31.988, "k3": 48.275, "k4": -7.201,
    "model": "radial_poly",
    "poly_order": 4,
    "width": 1280.0
  },
  "name": "FV"
}
```
Note the intrinsic keys are `cx_offset` / `cy_offset`, NOT `cx`/`cy`. The
`extrinsic.translation` + `extrinsic.quaternion` describe the transform from
camera coordinates to vehicle coordinates (ISO 8855 convention: X=forward,
Y=left, Z=up, origin at the midpoint of the rear axle).

## The algorithm (validated and working — see "what's already been proven" below)

Do NOT do the classic "undistort then homography" two-step approach. Instead,
sample directly:

1. Define a flat ground grid in vehicle coordinates around the car (e.g.
   X and Y both ranging ±10 meters, at Z=0), at your desired output resolution
   (e.g. 1000×1000 px).
2. For each of the 4 cameras, turn every grid point into a homogeneous
   `[X, Y, 0, 1]` vector and call `cam.project_3d_to_2d(world_points)` to get
   the fisheye pixel `(u, v)` each ground point corresponds to.
3. Use `cv2.remap` to sample the camera's actual image at those `(u, v)`
   coordinates — this produces one bird's-eye-view (BEV) patch per camera,
   already expressed in the same shared ground grid, so they're automatically
   spatially aligned with each other. Mark points with `u`/`v` = NaN or outside
   image bounds as invalid (alpha = 0).
4. Blend the 4 BEV patches into one canvas: weight each camera's contribution
   by a Gaussian-blurred version of its own valid-region mask (feathering), so
   overlapping corner regions blend smoothly instead of showing a hard seam.
   Sum `patch * weight` across cameras and divide by `sum(weight)`.
5. Save the result.

## What's already been proven to work on this project (don't re-derive)

- Loading `front.json` via `read_cam_from_json` and projecting a ground grid
  through `cam.project_3d_to_2d` on the repo's own `front.jpg` correctly
  flattens the road surface: lane markings become straight and parallel in
  the output, confirming the math and coordinate conventions are right.
- Known, expected artifact: anything NOT on the ground plane (parked cars,
  people, buildings, curbs) will smear/stretch radially in the BEV output,
  worse near the edges. This is a fundamental limitation of flat-ground
  projection, not a bug — don't spend time trying to "fix" it unless
  asked to explore learned/deep BEV methods instead.
- The 4-camera blend/canvas assembly code has been smoke-tested (with fake
  placeholder calibration for 3 of the 4 cameras) and runs without errors —
  the remaining unknown was only real calibration data + real synced images,
  which the user now has.
- A working reference implementation already exists — reuse and adapt it
  rather than starting over: `woodscape_surround_view.py` (attached /
  provided alongside this brief). It already:
  - imports `read_cam_from_json` from the cloned repo,
  - builds the ground grid,
  - projects + remaps each camera into a BEV patch,
  - feather-blends all 4 into one canvas,
  - takes `frame_id` and `data_dir` as the only things to configure.

## Task for you (the assistant working on this machine)

1. Find the dataset root and confirm the folder layout above.
2. Clone `valeoai/WoodScape` if not already present, to get `projection.py`.
3. Pick (or ask the user for) one frame index that has all 8 required files
   (4 images + 4 calibration jsons). List a few candidate indices if helpful.
4. Adapt `woodscape_surround_view.py`'s `WOODSCAPE_PROJECTION_DIR`, `frame_id`,
   and `data_dir` to the real paths on this machine.
5. Run it, inspect the output visually.
6. If you see doubled/ghosted objects at the seams between cameras — this is
   expected to some degree with WoodScape's stock extrinsics — mention it to
   the user rather than silently "fixing" it with arbitrary hacks. If they
   want it resolved properly, point them at `github.com/lwangvaleo/click_calib`
   ("Click-Calib"), a tool built specifically to refine WoodScape's extrinsic
   calibration for exactly this surround-view stitching use case by manually
   clicking corresponding ground points between adjacent camera pairs.
7. Optional nice-to-haves if the user wants to go further (don't do these
   unprompted, ask first): overlay a simple car silhouette icon in the middle
   of the canvas (the ground directly under the car is never observed by any
   camera); add photometric/exposure matching between cameras before blending,
   since each camera auto-exposes independently and seams may show a visible
   brightness jump even where geometry aligns perfectly.

## Style / working notes

- The user is new to computer vision concepts (fisheye projection, extrinsic/
  intrinsic calibration, bird's-eye-view synthesis) — when reporting back,
  explain results in plain language and tie visible artifacts back to
  the concepts above (e.g. "the smearing you see near the corners is the
  flat-ground assumption breaking down where a person is standing").
- Prefer showing the user the actual output image and describing what's
  correct/wrong about it over describing the code in the abstract.
