"""
Real-time pipeline configuration: which port carries which camera, where
each camera's (fixed) calibration lives, and the same projection/blend
settings validated in the batch pipeline (src/woodscape_surround_view.py).

This lane is fully separate from the batch/dataset pipeline in src/ and
scripts/ -- it only IMPORTS the reusable projection/blending code from
src/, it never edits it.
"""

import os

REALTIME_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REALTIME_DIR)

# Adjust this if your 4 streams arrive on different ports than assumed here.
CAMERA_PORTS = {
    "FV": 7011,
    "RV": 7012,
    "MVL": 7013,
    "MVR": 7014,
}

# Each camera's calibration is FIXED (a mounted camera's intrinsics/
# extrinsics don't change frame to frame the way the dataset's per-frame
# files implied) -- drop your real rig's 4 calibration JSONs here, same
# schema as WoodScape's (see data/woodscape/calibration_data/*.json).
CALIBRATION_DIR = os.path.join(REALTIME_DIR, "calibration")
CALIBRATION_PATHS = {cam: os.path.join(CALIBRATION_DIR, f"{cam}.json") for cam in CAMERA_PORTS}

# Same validated defaults as the batch pipeline's stitch_surround_view().
EXTENT_M = 6.0
RESOLUTION_PX = 1000
SURFACE = "bowl"
FLAT_RADIUS_M = 3.0
RIM_HEIGHT_M = 2.5
FV_INTRINSIC_SCALE = 0.79

# How stale a camera's last frame is allowed to be before that camera is
# treated as disconnected for this cycle (seconds).
MAX_FRAME_AGE_S = 1.0

TARGET_FPS = 10
