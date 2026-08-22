"""
Live version of the detection + ground-fusion + blind-spot pipeline
(src/experiment_fuse_detections.py, src/blind_spot_monitor.py): same
per-camera YOLO-seg detection, same mask-bottom ground-contact projection,
same blind-spot zone check -- driven by the 4 live GStreamer feeds instead
of dataset image files.

IMPORTANT environment note: this script needs BOTH an OpenCV build with
GStreamer support (system python3 has this) AND ultralytics/torch
(only installed in this repo's .venv, whose OpenCV build does NOT have
GStreamer). Neither environment alone can run this file as-is. Before
running it for real, either:
  - install ultralytics + torch into the system python3 environment, or
  - install an opencv-python build with GStreamer support into .venv
(whichever is easier on your actual deployment machine). This is a real
constraint of the two environments already set up in this repo, not a bug
in this script -- documented here rather than silently ignored.

    python3 realtime/realtime_detection.py

Same calibration/port setup as realtime_stitcher.py -- see its docstring.
Press 'q' in the display window (or Ctrl+C) to stop.
"""

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import YOLO_SEG_MODEL_PATH  # noqa: E402
from woodscape_surround_view import build_bowl_grid, camera_to_bev, blend_patches  # noqa: E402
from experiment_fuse_detections import (  # noqa: E402
    mask_ground_contact_pixel, pixel_to_ground, ground_to_canvas, draw_footprint,
    CAM_COLORS, MARKER_RADIUS_PX, DEFAULT_MARKER_RADIUS_PX, MIN_CONFIDENCE,
)
from blind_spot_monitor import check_zones, draw_zones, draw_warning, draw_banner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    CAMERA_PORTS, EXTENT_M, RESOLUTION_PX, FLAT_RADIUS_M, RIM_HEIGHT_M,
    MAX_FRAME_AGE_S, TARGET_FPS,
)
from gstreamer_capture import LiveCameraFeed, build_udp_h264_pipeline  # noqa: E402
from realtime_stitcher import load_cameras, start_feeds  # noqa: E402

CAMS = ["FV", "RV", "MVL", "MVR"]


def collect_live_detections(model, frames, cams):
    """Same logic as experiment_fuse_detections.collect_detections(), but
    reads already-decoded live frames (dict cam_name -> BGR ndarray)
    instead of loading dataset images by frame index."""
    detections = []
    for cam_name in CAMS:
        results = model(frames[cam_name], verbose=False)
        r = results[0]
        if r.boxes is None or r.masks is None:
            continue

        for mask_xy, cls_id, conf in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if len(mask_xy) < 3 or conf < MIN_CONFIDENCE:
                continue
            u, v = mask_ground_contact_pixel(mask_xy)
            ground = pixel_to_ground(cams[cam_name], u, v)
            if ground is None:
                continue
            gx, gy = ground
            if abs(gx) > EXTENT_M or abs(gy) > EXTENT_M:
                continue
            row, col = ground_to_canvas(gx, gy, EXTENT_M, RESOLUTION_PX)
            row, col = int(round(row)), int(round(col))
            if not (0 <= row < RESOLUTION_PX and 0 <= col < RESOLUTION_PX):
                continue

            detections.append({
                "cam": cam_name, "class": model.names[int(cls_id)], "conf": conf,
                "x": gx, "y": gy, "row": row, "col": col,
            })
    return detections


def main():
    from ultralytics import YOLO  # deferred: only needed here, see module docstring

    cams = load_cameras()
    model = YOLO(YOLO_SEG_MODEL_PATH)
    grid_x, grid_y, grid_z = build_bowl_grid(EXTENT_M, RESOLUTION_PX, FLAT_RADIUS_M, RIM_HEIGHT_M)
    feeds = start_feeds()

    frame_interval_s = 1.0 / TARGET_FPS
    try:
        while True:
            loop_start = time.time()

            frames, stale = {}, []
            for name, feed in feeds.items():
                frame, age = feed.get_latest_frame(max_age_s=MAX_FRAME_AGE_S)
                if frame is None:
                    stale.append(name)
                else:
                    frames[name] = frame

            if stale:
                print(f"waiting on: {stale} (no fresh frame within {MAX_FRAME_AGE_S}s)")
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, frame_interval_s - elapsed))
                continue

            patches, alphas = {}, {}
            for name in CAMS:
                patch = camera_to_bev(frames[name], cams[name], grid_x, grid_y, grid_z, cam_name=name)
                patches[name] = patch[:, :, :3]
                alphas[name] = patch[:, :, 3].astype("float32") / 255.0
            canvas = blend_patches(patches, alphas, EXTENT_M, RESOLUTION_PX)

            detections = collect_live_detections(model, frames, cams)
            draw_zones(canvas, lambda x, y: ground_to_canvas(x, y, EXTENT_M, RESOLUTION_PX))
            hits = check_zones(detections)

            for det in detections:
                radius_px = MARKER_RADIUS_PX.get(det["class"], DEFAULT_MARKER_RADIUS_PX)
                draw_footprint(canvas, det["row"], det["col"], radius_px, CAM_COLORS[det["cam"]],
                                f"{det['class']} {det['conf']:.2f}")
            for det, zone_name in hits:
                draw_warning(canvas, det, zone_name)
            draw_banner(canvas, hits)

            cv2.imshow("Live Surround View + Blind-Spot Monitor", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, frame_interval_s - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        for feed in feeds.values():
            feed.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
