"""
Pulls a live video feed from a GStreamer source into a background thread,
always exposing only the MOST RECENT decoded frame -- so a slow consumer
(the stitching loop) never builds up a backlog of stale frames; it just
always gets whatever is newest right now. This is a deliberate simplification
versus hardware-synchronized capture: the 4 cameras are not guaranteed to
be sampled at the exact same instant, only "recently" (see MAX_FRAME_AGE_S
in config.py). Good enough for a live monitoring display; not a substitute
for real hardware sync if this were feeding a safety-critical decision.

Requires OpenCV built with GStreamer support -- check with:
    python3 -c "import cv2; print(cv2.getBuildInformation())"
and look for "GStreamer: YES". On this machine, the system python3 has it;
.venv (built for the YOLO/torch side of this project) does not. Run
everything in realtime/ with the system python3, not .venv/bin/python3.
"""

import threading
import time

import cv2


def build_udp_h264_pipeline(port, latency_ms=100):
    """
    Default GStreamer pipeline for one camera: H.264-over-RTP arriving on
    a UDP port, decoded to BGR frames appsink hands to OpenCV.

    This is the one thing about a real camera rig this repo can't know in
    advance -- if your 4 streams are actually MJPEG, a different RTP
    payload type, TCP instead of UDP, etc., adjust the depay/decoder
    elements here (or pass your own pipeline string straight to
    LiveCameraFeed instead of using this helper).
    """
    return (
        f"udpsrc port={port} "
        f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
        f"! rtpjitterbuffer latency={latency_ms} "
        f"! rtph264depay ! h264parse ! avdec_h264 "
        f"! videoconvert ! appsink drop=true sync=false max-buffers=1"
    )


class LiveCameraFeed:
    """One camera's live feed. Call .start(), poll .get_latest_frame(),
    call .stop() when done."""

    def __init__(self, name, pipeline_str):
        self.name = name
        self.pipeline_str = pipeline_str
        self._cap = None
        self._frame = None
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self.pipeline_str, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[{self.name}] could not open GStreamer pipeline: {self.pipeline_str}\n"
                f"Check the port is correct, the stream is actually running, and that "
                f"this OpenCV build has GStreamer support (see module docstring)."
            )
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name=f"capture-{self.name}")
        self._thread.start()

    def _read_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_ts = time.time()

    def get_latest_frame(self, max_age_s=1.0):
        """Returns (frame, age_seconds), or (None, age_seconds_or_None) if
        nothing has arrived yet or the last frame is older than
        max_age_s (feed stalled or disconnected)."""
        with self._lock:
            if self._frame is None:
                return None, None
            age = time.time() - self._frame_ts
            if age > max_age_s:
                return None, age
            return self._frame.copy(), age

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
