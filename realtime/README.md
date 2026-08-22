# Real-Time Camera Pipeline

Live version of the batch dataset pipeline in `src/`/`scripts/`: instead of
reading pre-recorded WoodScape frames off disk by index, it pulls 4
continuous GStreamer video feeds and runs the exact same projection/blend
(and optionally detection + blind-spot) math on whatever's arriving right
now.

**This folder is fully separate from the batch/image pipeline** — nothing
here edits `src/` or `scripts/`. It only *imports* the reusable projection,
blending, detection-fusion, and blind-spot code from `src/`, so both lanes
stay in sync with exactly one implementation of the actual math; only
"where do frames come from" differs between them.

## Files

| File | Role |
|---|---|
| `config.py` | Port-to-camera mapping, calibration paths, projection/blend settings (mirrors the validated batch-pipeline defaults) |
| `gstreamer_capture.py` | `LiveCameraFeed` — opens one GStreamer source in a background thread, always exposes the latest decoded frame |
| `realtime_stitcher.py` | Stitching only: 4 feeds → live surround view, no detection |
| `realtime_detection.py` | Stitching + YOLO detection + ground fusion + blind-spot warnings, live |
| `calibration/` | Drop your camera rig's real calibration JSONs here (empty until you do) |

## Setup

**1. Calibration.** Unlike the dataset (a different calibration file per
frame index, because it's simulating many different recording sessions),
a real camera rig's calibration is fixed — it doesn't change frame to
frame. Put exactly 4 files here, same JSON schema WoodScape uses (see
`data/woodscape/calibration_data/*.json` for the format):

```
realtime/calibration/FV.json
realtime/calibration/RV.json
realtime/calibration/MVL.json
realtime/calibration/MVR.json
```

**2. Ports.** `config.py`'s `CAMERA_PORTS` assumes `FV=7011, RV=7012,
MVL=7013, MVR=7014` in that order — edit that dict if your 4 streams are
numbered differently.

**3. Stream format.** `gstreamer_capture.build_udp_h264_pipeline()`
assumes H.264-over-RTP arriving on a UDP port. This is the one thing about
a real camera rig this repo can't know in advance — if your streams are
actually MJPEG, a different RTP payload type, TCP instead of UDP, etc.,
edit that pipeline string (or build your own and pass it straight to
`LiveCameraFeed`).

**4. Environment.** `realtime_stitcher.py` needs an OpenCV build with
GStreamer support — check with:

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())"   # look for "GStreamer: YES"
```

On the machine this was developed on, the **system python3** has this;
the repo's `.venv` (built for the YOLO/torch detection side of the
project) does not. `realtime_detection.py` needs *both* GStreamer-enabled
OpenCV *and* `ultralytics`/`torch` in the same environment — neither of
this repo's two existing environments has both, so before running it for
real you'll need to either install `ultralytics`+`torch` into the system
python3, or install a GStreamer-enabled `opencv-python` build into
`.venv`. This is a genuine constraint of how the two environments are
currently split, not a bug in the script.

## Running it

```bash
# Stitching only, no detection
python3 realtime/realtime_stitcher.py

# Stitching + detection + blind-spot warnings (needs both environments
# merged -- see "Environment" above)
python3 realtime/realtime_detection.py
```

Press `q` in the display window, or Ctrl+C in the terminal, to stop.

## Known limitations / simplifications

- **No hardware-synchronized capture.** Each camera's background thread
  always exposes whatever frame arrived most recently; the main loop
  reads all 4 "current" frames when it's ready to stitch. Frames across
  cameras are only guaranteed to be *recent* (within `MAX_FRAME_AGE_S`,
  default 1s), not sampled at the exact same instant the way the
  dataset's timestamp-matched quadruples are. Good enough for a live
  monitoring display; not a substitute for real hardware sync in a
  safety-critical system.
- **Best-effort framerate**, not a hard real-time guarantee — `TARGET_FPS`
  paces the loop but a slow stitch/detect cycle just makes the next frame
  later, it doesn't drop frames or reorder anything.
- Everything else — off-ground-plane smearing, the value of detecting on
  raw frames rather than the stitched canvas, blind-spot zone limitations
  — is identical to the batch pipeline's, since it's the same code. See
  `docs/surround_view_pipeline.md` and `docs/blind_spot_monitoring.md`.
