"""
Live version of the batch stitching pipeline (src/woodscape_surround_view.py):
same ground-grid projection + blend math (build_bowl_grid, camera_to_bev,
blend_patches -- imported, not duplicated), driven by 4 continuous
GStreamer feeds instead of 4 dataset image files.

Run with the SYSTEM python3 (needs an OpenCV build with GStreamer support --
see gstreamer_capture.py's docstring):

    python3 realtime/realtime_stitcher.py

Before running: drop your camera rig's real calibration JSONs into
realtime/calibration/{FV,RV,MVL,MVR}.json (same schema as WoodScape's --
see data/woodscape/calibration_data/*.json for the format), and check
CAMERA_PORTS in config.py matches how your 4 streams are actually numbered.

Press 'q' in the display window (or Ctrl+C in the terminal) to stop.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from woodscape_surround_view import (  # noqa: E402
    build_bowl_grid, camera_to_bev, blend_patches, read_cam_from_json,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    CAMERA_PORTS, CALIBRATION_PATHS, EXTENT_M, RESOLUTION_PX,
    FLAT_RADIUS_M, RIM_HEIGHT_M, FV_INTRINSIC_SCALE, MAX_FRAME_AGE_S, TARGET_FPS,
)
from gstreamer_capture import LiveCameraFeed, build_udp_h264_pipeline  # noqa: E402

CAMS = ["FV", "RV", "MVL", "MVR"]


def load_cameras():
    """Loads each camera's fixed calibration once at startup (unlike the
    dataset pipeline, a live rig's calibration doesn't change per frame)."""
    cams = {}
    for name in CAMS:
        path = CALIBRATION_PATHS[name]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing calibration for {name}: {path}\n"
                f"Drop your camera rig's real calibration JSON there first "
                f"(same schema as WoodScape's -- see "
                f"data/woodscape/calibration_data/*.json)."
            )
        cam = read_cam_from_json(path)
        if name == "FV":
            cam.lens.coefficients = cam.lens.coefficients * FV_INTRINSIC_SCALE
        cams[name] = cam
    return cams


def start_feeds():
    feeds = {}
    for name, port in CAMERA_PORTS.items():
        feed = LiveCameraFeed(name, build_udp_h264_pipeline(port))
        feed.start()
        feeds[name] = feed
        print(f"[{name}] listening on UDP port {port}")
    return feeds


def stitch_one_frame(frames, cams, grid_x, grid_y, grid_z):
    """frames: dict cam_name -> BGR ndarray (already-decoded live frame).
    Same camera_to_bev() + blend_patches() the batch pipeline uses --
    the only new thing here is that `frames` came from a live feed."""
    patches, alphas = {}, {}
    for name in CAMS:
        patch = camera_to_bev(frames[name], cams[name], grid_x, grid_y, grid_z, cam_name=name)
        patches[name] = patch[:, :, :3]
        alphas[name] = patch[:, :, 3].astype(np.float32) / 255.0
    return blend_patches(patches, alphas, EXTENT_M, RESOLUTION_PX)


def main():
    cams = load_cameras()
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
            else:
                result = stitch_one_frame(frames, cams, grid_x, grid_y, grid_z)
                cv2.imshow("Live Surround View", result)
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
