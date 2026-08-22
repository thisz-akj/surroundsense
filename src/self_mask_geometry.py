"""
Static ego-vehicle self-occlusion mask, defined geometrically per camera
rather than estimated from image statistics.

Why not estimate it from images: per-pixel variance across many frames
only catches things that are BOTH spatially fixed AND visually identical
every time (e.g. the fisheye's own dead corners). The vehicle's own paint
and chrome trim are spatially fixed but visually change with ambient
light and reflections, so a pure statistics test misses most of it (tried
first, see build_self_masks.py -- it only ever caught a sliver).

What actually works, and is standard practice: this boundary never moves
because the camera is bolted to the car. It only needs to be measured
once, by eye, from the raw image (or its across-many-frames MEAN, which
is what was used here -- persistent text/edges like the rear license
plate frame survive averaging crisply while the ever-changing scene blurs
into a smooth gradient, making the true boundary easy to read off).

FV, RV: the bumper trim's highest point across the frame is used as a
flat cutoff row (slightly conservative -- some usable near-field ground
is sacrificed rather than risk leaving a sliver of chrome visible).
MVL, MVR: the car's own door/body fills a diagonal wedge instead of a
horizontal band, so the cutoff is a straight line, not a row.
"""

import numpy as np

# row cutoff for FV/RV: everything with v >= this row is masked
_ROW_CUTOFF = {"FV": 650, "RV": 560}

# (slope, intercept) for MVL/MVR: masked where v >= slope*u + intercept
_LINE_CUTOFF = {
    "MVL": (0.486, 300),
    "MVR": (-0.486, 860),
}


def self_mask_valid(u, v, cam_name):
    """
    u, v: pixel coordinate arrays (any shape, matching).
    Returns a boolean array, True where the pixel shows the outside world
    (keep), False where it shows the vehicle's own body (mask out).
    """
    if cam_name in _ROW_CUTOFF:
        return v < _ROW_CUTOFF[cam_name]
    slope, intercept = _LINE_CUTOFF[cam_name]
    return v < (slope * u + intercept)
