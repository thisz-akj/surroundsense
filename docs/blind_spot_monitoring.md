# Blind-Spot Monitoring — What We Built and How It Works

This documents the blind-spot warning system built on top of the surround-view
stitching pipeline (see `docs/surround_view_pipeline.md`). It answers three
things the user asked for: detect vehicles/people, define blind-spot zones,
and trigger a warning when something enters one.

## 1. The big idea

A "blind spot" is a region beside and behind the car that's hard for a
*driver* to see through mirrors — it has nothing to do with a gap in our
own camera coverage (our 4 cameras already see all the way around the
car). So the system doesn't need a 5th camera or a new sensor; it needs to:

1. Find real-world objects using the cameras we already have.
2. Know exactly where they are in **vehicle-frame meters**, not just which
   pixel they're in.
3. Check that position against a couple of hand-defined rectangles that
   represent where a real BSM (blind-spot monitoring) icon would light up
   in a normal car.

Step 2 is the one piece of real geometry in this whole feature — everything
else is comparison against fixed numbers.

## 2. Why detection happens on the raw camera, not the stitched image

Covered in depth in the earlier ADAS discussion and `surround_view_pipeline.md`
§5, but the short version: our stitched BEV image assumes everything is
flat on the ground (Z=0), so any real object with height gets smeared once
it's forced onto that flat canvas. A detector trained on ordinary photos
fails badly on that smeared shape (we measured this: only ~50% of stitched
detections were even plausible object classes, vs. 97% when the same model
ran on the raw, undistorted-by-our-assumption camera images). So detection
always runs on the **raw fisheye image**, per camera, and only the
*resulting position* gets combined afterward — never the pixels.

## 3. Architecture

```
raw FV/RV/MVL/MVR image
        |
        v
YOLO11n-seg (pretrained, COCO)  --> class, confidence, segmentation mask
        |
        v
mask_ground_contact_pixel()     --> the mask's own lowest pixels (u, v)
        |
        v
pixel_to_ground()                --> back-project through that camera's
                                      OWN calibration to find where the
                                      ray hits Z=0: real (X, Y) meters
        |
        v
check_zones()                    --> is (X, Y) inside a blind-spot zone?
        |
        v
draw_warning() / draw_banner()   --> red highlight + "WARNING: ..." banner
```

Each stage is a real, separately-tested piece — not a black box:

- **`mask_ground_contact_pixel`** (`src/experiment_fuse_detections.py`) uses
  the segmentation mask's own lowest edge, not the bounding box's
  bottom-corner. We tried the bbox version first; it put markers
  noticeably off from the real object once photographed at an angle
  (common in fisheye/mirror views), because a box corner can sit outside
  the actual silhouette. The mask's true bottom is a much closer match to
  where the object touches the ground.

- **`pixel_to_ground`** back-projects a fisheye pixel to a 3D ray using
  the *same* calibration model (`Camera.project_2d_to_3d`) the stitching
  pipeline already relies on, then solves for where that ray crosses
  Z=0 — pure ray/plane intersection, no new math introduced for this
  feature. We verified this against the actual stitched canvas pixels
  (cropped the canvas at a few computed positions and confirmed real
  car/person content sits right there) before trusting it for zone checks.

- **Confidence filter** (`MIN_CONFIDENCE = 0.30`): detections below this
  are dropped before they ever reach the zone check, cutting down on the
  low-confidence noise (occasional "suitcase"/misread classes) that showed
  up in the raw per-camera detection sweep.

## 4. The blind-spot zones

Defined in `src/blind_spot_monitor.py`, in vehicle-frame meters (X=forward,
Y=left, origin=vehicle center — matching `draw_car_silhouette`'s car body):

```python
CAR_WIDTH_M = 1.8
BLIND_SPOT_ZONES = {
    "left":  (-4.0, 1.0,  CAR_WIDTH_M/2,        CAR_WIDTH_M/2 + 3.0),
    "right": (-4.0, 1.0, -(CAR_WIDTH_M/2 + 3.0), -CAR_WIDTH_M/2),
}
# each tuple is (x_min, x_max, y_min, y_max)
```

Each zone runs from a bit ahead of the front bumper back through and past
the rear bumper (X from -4.0 to +1.0), offset sideways from the car's own
edge out to 3 more meters (Y band) — the same footprint a real BSM icon
lights up for a car in the next lane, or a cyclist/pedestrian coming up
from behind at the side. `point_in_zone()` is a plain rectangle
containment check; `check_zones()` runs it against every detection this
frame and returns every (detection, zone) pair that's inside.

These numbers are a reasonable default, not a tuned spec — they're easy to
widen/narrow in one place if a particular vehicle geometry or use case
calls for it.

## 5. What gets drawn

- **Zone outlines** (`draw_zones`): semi-transparent amber rectangles,
  projected onto the canvas using the exact same `ground_to_canvas()`
  function the detections themselves were projected with — so the zone
  and anything that's supposed to trigger it share one coordinate mapping,
  not two that could silently drift apart.
- **Normal detections**: the same small filled, class-colored dot from the
  earlier fusion work (car/bus/person/etc., sized per class, colored by
  which camera saw it).
- **Triggering detections** (`draw_warning`): circled in red with a
  `BLIND SPOT (side): class` label right next to the marker.
- **Top banner** (`draw_banner`): a one-line summary across the top of the
  frame — green "BLIND SPOT: clear" or red "WARNING: bicycle (left),
  person (right)" — the single thing a driver display would actually show.

## 6. Result on the 50-frame sample

22 of 50 frames triggered at least one warning — mostly cyclists and
pedestrians approaching from the rear-left or rear-right (busy bike-parking
scenes like idx0353 correctly flag several people and bicycles across both
zones at once). Full outputs, per-frame images, and the warning log are in
`outputs/experiments/blind_spot_demo_50/` (`idxNNNN.png`, `warnings.json`,
`contact_sheet.png`).

## 7. Known limitations

- **Occasional false triggers from the ego vehicle's own body.** The raw-
  camera model sometimes misreads the car's own hood or mirror as a
  nearby object — a detection-model issue carried over from the earlier
  fusion work, not something the zone-check logic itself can fix.
- **No heading or velocity.** Detections come from a single frame each;
  there's no tracking across frames, so a stationary parked car and an
  oncoming cyclist look identical to the zone check (both are just "an
  object is inside this rectangle right now"). Real BSM systems track
  objects over time and often only warn on approaching/closing objects.
- **Static per-frame demo, not a live feed.** This runs on 50 discrete
  synced samples, not a continuous video stream — there's no frame-to-frame
  smoothing, so a single missed or spurious detection isn't damped by
  neighboring frames the way a real deployed system would.
- **Zones are hand-set rectangles, not derived from the vehicle's actual
  dimensions file** (if WoodScape ships one) or from mirror
  field-of-view geometry — they're a reasonable approximation, not a
  calibrated spec for this specific vehicle.

## 8. File map

| File | Role |
|---|---|
| `src/blind_spot_monitor.py` | Zone definitions, `point_in_zone`/`check_zones`, `draw_zones`/`draw_warning`/`draw_banner` |
| `src/experiment_fuse_detections.py` | `collect_detections()` — shared per-camera detect + mask-based ground-fusion, used by both the plain fusion demo and the blind-spot demo |
| `scripts/experiment_blind_spot_demo.py` | Runs the full pipeline over `data/sample_50.json` and writes annotated frames + `warnings.json` |
| `outputs/experiments/blind_spot_demo_50/` | Output: per-frame annotated PNGs, warning log, contact sheet |
