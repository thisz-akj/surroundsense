"""
Every path this project cares about, computed relative to the repo root
instead of hardcoded to one machine's home directory -- so cloning this
repo anywhere still works. Every script imports from here rather than
hardcoding its own copy of these strings.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WOODSCAPE_DATA_DIR = os.path.join(DATA_DIR, "woodscape")
IMG_DIR = os.path.join(WOODSCAPE_DATA_DIR, "rgb_images")
CALIB_DIR = os.path.join(WOODSCAPE_DATA_DIR, "calibration_data")
VEHICLE_DATA_DIR = os.path.join(WOODSCAPE_DATA_DIR, "vehicle_data")

MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
SAMPLE_10_PATH = os.path.join(DATA_DIR, "sample_10.json")
SAMPLE_20_PATH = os.path.join(DATA_DIR, "sample_20.json")
SAMPLE_50_PATH = os.path.join(DATA_DIR, "sample_50.json")

EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")
WOODSCAPE_REPO_DIR = os.path.join(EXTERNAL_DIR, "WoodScape")
WOODSCAPE_PROJECTION_DIR = os.path.join(WOODSCAPE_REPO_DIR, "scripts", "calibration")
CLICK_CALIB_DIR = os.path.join(EXTERNAL_DIR, "click_calib")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
YOLO_SEG_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n-seg.pt")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPERIMENTS_DIR = os.path.join(OUTPUTS_DIR, "experiments")
SELF_MASKS_DIR = os.path.join(OUTPUTS_DIR, "self_masks")
SURROUND_VIEWS_DIR = os.path.join(OUTPUTS_DIR, "surround_views")

# docs/ is committed (unlike outputs/) -- finished, curated deliverables
# meant to ship with the repo (PDF reports, README images) live here.
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DOCS_IMAGES_DIR = os.path.join(DOCS_DIR, "images")
