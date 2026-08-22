"""
Batch-generate surround-view (BEV) images for every genuinely time-synchronized
4-camera quadruple in the dataset.

"Synchronized" here means all 4 cameras' vehicle_data/rgb_images/*.json report
the EXACT SAME `timestamp` value (spread = 0) -- the strongest sync guarantee
this dataset can give, found because filename indices are NOT synchronized
across cameras (see woodscape_surround_view.py's docstring for why).

Outputs:
- surround_views/sv_<NNN>_FV<fv>_RV<rv>_MVL<mvl>_MVR<mvr>.png  (one per quadruple)
- manifest.json  (list of {timestamp, FV, RV, MVL, MVR, output} records)
"""

import os
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import IMG_DIR, CALIB_DIR, VEHICLE_DATA_DIR as VEHICLE_DATA_ROOT, SURROUND_VIEWS_DIR, MANIFEST_PATH  # noqa: E402
from woodscape_surround_view import stitch_surround_view  # noqa: E402

VEHICLE_DATA_DIR = os.path.join(VEHICLE_DATA_ROOT, "rgb_images")
OUT_DIR = SURROUND_VIEWS_DIR

CAMS = ["FV", "RV", "MVL", "MVR"]


def find_synced_quadruples():
    """Group all vehicle_data JSONs by exact timestamp; keep only timestamps
    where each of the 4 cameras has exactly one frame."""
    cam_by_ts = defaultdict(lambda: defaultdict(list))
    for f in os.listdir(VEHICLE_DATA_DIR):
        idx, cam = f[:-5].split("_")
        with open(os.path.join(VEHICLE_DATA_DIR, f)) as fh:
            data = json.load(fh)
        ts = int(data["timestamp"])
        cam_by_ts[ts][cam].append(idx)

    quads = []
    for ts, campair in cam_by_ts.items():
        if all(c in campair for c in CAMS) and all(len(campair[c]) == 1 for c in CAMS):
            quads.append((ts, {c: campair[c][0] for c in CAMS}))

    quads.sort(key=lambda x: x[0])
    return quads


def verify_files_exist(frame_ids):
    for c in CAMS:
        idx = frame_ids[c]
        if not os.path.exists(f"{IMG_DIR}/{idx}_{c}.png"):
            return False
        if not os.path.exists(f"{CALIB_DIR}/{idx}_{c}.json"):
            return False
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    quads = find_synced_quadruples()
    quads = [(ts, fids) for ts, fids in quads if verify_files_exist(fids)]
    print(f"Found {len(quads)} verified synchronized quadruples. Stitching...")

    manifest = []
    t_start = time.time()
    for i, (ts, frame_ids) in enumerate(quads):
        out_name = (
            f"sv_{i:04d}_FV{frame_ids['FV']}_RV{frame_ids['RV']}"
            f"_MVL{frame_ids['MVL']}_MVR{frame_ids['MVR']}.png"
        )
        out_path = os.path.join(OUT_DIR, out_name)

        import cv2
        surround = stitch_surround_view(frame_ids, IMG_DIR, CALIB_DIR, extent_m=6.0, resolution_px=1000)
        cv2.imwrite(out_path, surround)

        manifest.append({
            "index": i,
            "timestamp": ts,
            "FV": frame_ids["FV"],
            "RV": frame_ids["RV"],
            "MVL": frame_ids["MVL"],
            "MVR": frame_ids["MVR"],
            "output": out_name,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(quads):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(quads) - (i + 1)) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(quads)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Done. {len(manifest)} surround views written to {OUT_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
