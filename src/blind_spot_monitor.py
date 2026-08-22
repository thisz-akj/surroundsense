"""
Blind-spot monitoring on top of the existing detect-per-camera ->
fuse-to-ground-frame pipeline (experiment_fuse_detections.collect_detections).

A "blind spot" here means the classic BSM (blind-spot monitoring) zone: a
region alongside and just behind the car that's hard for a driver to see
via mirrors -- NOT a gap in our own camera coverage (our 4 cameras already
see all the way around). We already have every detected object's real
ground position in vehicle-frame meters (see collect_detections); this
module just asks "is that position inside a zone we care about."

Zone convention matches draw_car_silhouette()'s car body in
woodscape_surround_view.py: centered on the vehicle-frame origin,
CAR_LENGTH_M x CAR_WIDTH_M.
"""

import cv2

CAR_LENGTH_M = 4.3
CAR_WIDTH_M = 1.8

# (x_min, x_max, y_min, y_max) in vehicle-frame meters (X=forward, Y=left).
# Each zone runs from a bit ahead of the front bumper back through and
# past the rear bumper, offset outward from the car's own side -- the
# same footprint a real BSM icon lights up for a car sitting in the next
# lane over, or a cyclist/pedestrian approaching from behind at the side.
BLIND_SPOT_ZONES = {
    "left": (-4.0, 1.0, CAR_WIDTH_M / 2, CAR_WIDTH_M / 2 + 3.0),
    "right": (-4.0, 1.0, -(CAR_WIDTH_M / 2 + 3.0), -CAR_WIDTH_M / 2),
}

ZONE_COLOR = (0, 165, 255)   # amber outline for the zone itself
WARNING_COLOR = (0, 0, 255)  # red highlight for a triggering detection


def point_in_zone(x, y, zone):
    x_min, x_max, y_min, y_max = zone
    return x_min <= x <= x_max and y_min <= y <= y_max


def check_zones(detections, zones=BLIND_SPOT_ZONES):
    """detections: list of dicts with at least 'x','y' (vehicle-frame
    meters), as produced by collect_detections(). Returns a list of
    (detection, zone_name) for every detection currently inside a zone."""
    hits = []
    for det in detections:
        for zone_name, zone in zones.items():
            if point_in_zone(det["x"], det["y"], zone):
                hits.append((det, zone_name))
    return hits


def _zone_polygon_px(zone, ground_to_canvas):
    """Converts a zone's 4 ground-meter corners to (col, row) canvas pixel
    points, in the (x,y) order cv2 expects (col=x, row=y)."""
    import numpy as np
    x_min, x_max, y_min, y_max = zone
    corners_m = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    pts = []
    for x, y in corners_m:
        row, col = ground_to_canvas(x, y)
        pts.append([int(round(col)), int(round(row))])
    return np.array(pts, dtype=np.int32)


def draw_zones(canvas, ground_to_canvas, zones=BLIND_SPOT_ZONES, color=ZONE_COLOR, alpha=0.18):
    """Draws each zone as a semi-transparent rectangle on the canvas, using
    the same ground_to_canvas(x, y) function the detections were projected
    with, so the zone and the detections it's supposed to catch share
    exactly one coordinate mapping."""
    overlay = canvas.copy()
    for zone_name, zone in zones.items():
        cv2.fillPoly(overlay, [_zone_polygon_px(zone, ground_to_canvas)], color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

    for zone_name, zone in zones.items():
        poly = _zone_polygon_px(zone, ground_to_canvas)
        cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=2)
        label_col, label_row = poly[2]  # (x_max, y_max) corner
        cv2.putText(canvas, zone_name.upper(), (label_col - 55, label_row - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_warning(canvas, det, zone_name):
    """Highlights a triggering detection in red and draws a warning banner
    naming what/where."""
    cv2.circle(canvas, (det["col"], det["row"]), 22, WARNING_COLOR, thickness=3)
    label = f"BLIND SPOT ({zone_name}): {det['class']}"
    cv2.putText(canvas, label, (det["col"] - 60, det["row"] - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WARNING_COLOR, 1, cv2.LINE_AA)


def draw_banner(canvas, hits):
    """Top-of-frame summary banner, red if anything is currently triggering."""
    if not hits:
        text = "BLIND SPOT: clear"
        color = (80, 200, 80)
    else:
        names = ", ".join(sorted({f"{d['class']} ({z})" for d, z in hits}))
        text = f"WARNING: {names}"
        color = WARNING_COLOR
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (20, 20, 20), thickness=-1)
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
