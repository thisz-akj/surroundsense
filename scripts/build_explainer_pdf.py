"""
Builds a diagram-rich, beginner-friendly PDF walking through the whole
surround-view project: raw fisheye files -> final stitched output.
Every page is a simple schematic (boxes, arrows, curves) rather than a
real photo, on purpose -- the goal is to make the underlying idea click,
not to be visually fancy.

Pages are standard portrait A4. The data coordinate system is (0-100)
in X and (0-DATA_H) in Y, where DATA_H is chosen so 1 data-unit is the
same physical size in both directions (no stretched circles/wedges).
Box/header text auto-shrinks to fit its container -- portrait A4 is
narrower than the landscape draft this was first built at, so anything
sized by eye for the old width silently overflowed.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch, Wedge, Polygon
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import DOCS_DIR  # noqa: E402

REPORT_PATH = os.path.join(DOCS_DIR, "surround_view_explainer.pdf")

PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches
DATA_W = 100.0
DATA_H = DATA_W * PAGE_SIZE[1] / PAGE_SIZE[0]  # ~141.35, keeps circles circular
PT_PER_UNIT_X = PAGE_SIZE[0] * 72 / DATA_W

NAVY = "#1c2b4a"
BLUE = "#2f6fb0"
TEAL = "#2f9e8f"
ORANGE = "#e07b39"
RED = "#c0392b"
GREY = "#8a8f98"
LIGHT = "#eef1f6"
GREEN = "#3f9142"
PURPLE = "#7d3c98"

HEADER_STEP_Y = DATA_H - 7
HEADER_TITLE_Y = DATA_H - 16
HEADER_SUB_Y = DATA_H - 23.5
HEADER_RULE_Y = DATA_H - 27.5
CONTENT_TOP = DATA_H - 34
CONTENT_BOTTOM = 11
FOOTER_Y = 5

plt.rcParams["font.family"] = "DejaVu Sans"


def _fit_fontsize(text, width_units, requested_fs, char_w=0.60, pad_frac=0.86):
    longest = max((len(l) for l in text.split("\n") if l.strip()), default=1)
    avail_pt = width_units * PT_PER_UNIT_X * pad_frac
    max_fs = avail_pt / max(longest * char_w, 1)
    return min(requested_fs, max_fs)


def new_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, DATA_W)
    ax.set_ylim(0, DATA_H)
    ax.axis("off")
    return fig, ax


def header(ax, step, title, subtitle=None):
    ax.text(4, HEADER_STEP_Y, step, fontsize=13, color=ORANGE, weight="bold")
    fs = _fit_fontsize(title, DATA_W - 8, 20)
    ax.text(4, HEADER_TITLE_Y, title, fontsize=fs, color=NAVY, weight="bold")
    if subtitle:
        fs_sub = _fit_fontsize(subtitle, DATA_W - 8, 11.5, char_w=0.52)
        ax.text(4, HEADER_SUB_Y, subtitle, fontsize=fs_sub, color="#333333", style="italic")
    ax.plot([4, DATA_W - 4], [HEADER_RULE_Y, HEADER_RULE_Y], color=LIGHT, lw=2)


def footer(ax, note):
    fs = _fit_fontsize(note, DATA_W - 16, 9, char_w=0.50)
    ax.text(DATA_W / 2, FOOTER_Y, note, fontsize=fs, color=GREY, ha="center", style="italic")


def box(ax, x, y, w, h, text, color=BLUE, fontsize=11, text_color="white", alpha=1.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                        linewidth=0, facecolor=color, alpha=alpha)
    ax.add_patch(b)
    fs = _fit_fontsize(text, w, fontsize)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, weight="bold")
    return b


def caption(ax, x, y, text, fontsize=11, color="#333333", **kw):
    fs = _fit_fontsize(text, DATA_W - 2 * x if kw.get("ha", "center") == "center" else DATA_W - x,
                        fontsize, char_w=0.52)
    ax.text(x, y, text, fontsize=fs, color=color, **kw)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                         color=color, lw=lw, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def camera_icon(ax, x, y, r=1.6, color=NAVY, label=None, label_dy=-3.2):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", lw=1.2, zorder=5))
    ax.add_patch(Circle((x, y), r * 0.45, facecolor="white", zorder=6))
    if label:
        ax.text(x, y + label_dy, label, ha="center", fontsize=10, color=color, weight="bold")


def fov_wedge(ax, x, y, direction_deg, spread_deg, radius, color, alpha=0.18):
    w = Wedge((x, y), radius, direction_deg - spread_deg / 2, direction_deg + spread_deg / 2,
              facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
    ax.add_patch(w)


def car_topdown(ax, cx, cy, length=14, width=7, color="#444444"):
    body = FancyBboxPatch((cx - width / 2, cy - length / 2), width, length,
                           boxstyle="round,pad=0,rounding_size=2.2",
                           facecolor=color, edgecolor="white", lw=1)
    ax.add_patch(body)
    ax.add_patch(Rectangle((cx - width / 2 + 1, cy + length / 2 - 4.5), width - 2, 2.2,
                            facecolor="#7fb3d5", edgecolor="none", alpha=0.6))


PP = PdfPages(REPORT_PATH)


# ============================================================ PAGE 1: TITLE
fig, ax = new_page()
ax.text(50, DATA_H - 26, "Building a 360° Surround View", fontsize=25, color=NAVY,
        weight="bold", ha="center")
ax.text(50, DATA_H - 35, "from 4 Fisheye Cameras", fontsize=25, color=NAVY, weight="bold", ha="center")
ax.text(50, DATA_H - 46, "A beginner's tour through everything we did — and why —", fontsize=13,
        color="#333333", ha="center", style="italic")
ax.text(50, DATA_H - 51.5, "on the way from raw sensor files to a clean top-down view.", fontsize=13,
        color="#333333", ha="center", style="italic")

steps = [
    "1. The setup: 4 cameras, 1 car",
    "2. How a fisheye lens sees the world",
    "3. The core trick: ask backwards, don't warp",
    "4. Blending 4 views into 1 picture",
    "5. Fixing frames that weren't really synchronized",
    "6. Why far-away things looked smeared and blocky",
    "7. Matching brightness across cameras",
    "8. A dead end, investigated properly (feature matching)",
    "9. The two feature-matching techniques, in detail",
    "10. Cleaning up sensor grain",
    "11. Removing the car's own reflection from its view",
    "12. The 'bowl' trick used by real car displays",
    "13. Trying real depth estimation (and why it broke)",
    "14. Fixing double images at the seams",
]
y = DATA_H - 64
for s in steps:
    ax.text(14, y, s, fontsize=12, color=NAVY)
    y -= 4.6
footer(ax, "Every page after this one explains ONE of these steps with a simple diagram.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 2: STEP 1
fig, ax = new_page()
header(ax, "STEP 1", "Four Eyes Around a Car",
       "Each camera comes with a file describing exactly where it is and which way it looks")

ax.text(6, CONTENT_TOP, "Calibration file =", fontsize=13, color=NAVY, weight="bold")
ax.text(6, CONTENT_TOP - 6, "EXTRINSIC (where it is + which way it points)", fontsize=11, color=BLUE)
ax.text(6, CONTENT_TOP - 11, "+", fontsize=11, color="#555555")
ax.text(6, CONTENT_TOP - 16, "INTRINSIC (how the lens itself bends light)", fontsize=11, color=TEAL)

diagram_cy = CONTENT_TOP - 62
car_topdown(ax, 50, diagram_cy, length=20, width=10)
cams = [
    (50, diagram_cy + 12, 90, "FV\n(front)"),
    (50, diagram_cy - 12, 270, "RV\n(rear)"),
    (40, diagram_cy, 180, "MVL\n(left mirror)"),
    (60, diagram_cy, 0, "MVR\n(right mirror)"),
]
colors = [BLUE, RED, TEAL, ORANGE]
for (x, y, ang, label), c in zip(cams, colors):
    fov_wedge(ax, x, y, ang, 170, 23, c, alpha=0.15)
    camera_icon(ax, x, y, r=1.3, color=c)
    if x == 50:
        ax.text(x, y + (9.5 if y > diagram_cy else -8.5), label, ha="center", fontsize=10.5,
                color=c, weight="bold")
    else:
        ax.text(x + (7 if x > 50 else -7), y, label, ha="center", fontsize=10.5,
                color=c, weight="bold")

ax.text(50, diagram_cy - 46, "The shaded fans show roughly what each camera can see (~180° each) —\n"
                             "notice the corners where two cameras' views overlap. That overlap is\n"
                             "where most of our later problems (and fixes) happen.",
        ha="center", fontsize=11.5, color="#333333")
footer(ax, "We reused Valeo's own calibration math instead of re-deriving fisheye geometry from scratch.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 3: STEP 2
fig, ax = new_page()
header(ax, "STEP 2", "How a Fisheye Lens Sees the World",
       "Why an ordinary camera model can't describe a ~180° lens")

label_y = CONTENT_TOP
ax.text(25, label_y, "Ordinary (pinhole) lens", fontsize=13, color=NAVY, weight="bold", ha="center")
ax.text(75, label_y, "Fisheye lens", fontsize=13, color=NAVY, weight="bold", ha="center")

plot_h_units = 42
plot_bottom_units = label_y - 14 - plot_h_units
plot_bottom_frac = plot_bottom_units / DATA_H
plot_h_frac = plot_h_units / DATA_H
ax_l = plt.axes([0.10, plot_bottom_frac, 0.36, plot_h_frac])
theta = np.linspace(0, 1.45, 200)
ax_l.plot(np.degrees(theta), np.tan(theta), color=RED, lw=2.5)
ax_l.set_ylim(0, 8)
ax_l.set_xlim(0, 90)
ax_l.axvline(83, color=GREY, ls="--", lw=1)
ax_l.text(6, 6.6, "pixel distance shoots\nto infinity as angle → 90°", fontsize=9, color=RED)
ax_l.set_xlabel("angle from center (°)", fontsize=9.5)
ax_l.set_ylabel("distance from image\ncenter (pixels, scaled)", fontsize=9.5)
ax_l.set_title("pixel_distance ∝ tan(angle)", fontsize=10.5, color=RED)
ax_l.tick_params(labelsize=8.5)

ax_r = plt.axes([0.58, plot_bottom_frac, 0.36, plot_h_frac])
theta2 = np.linspace(0, 3.0, 200)
ax_r.plot(np.degrees(theta2), 1 - np.exp(-theta2 * 0.9), color=TEAL, lw=2.5)
ax_r.set_ylim(0, 1.1)
ax_r.set_xlim(0, 170)
ax_r.text(15, 0.15, "stays bounded even past\n90°, all the way to ~180°", fontsize=9, color=TEAL)
ax_r.set_xlabel("angle from center (°)", fontsize=9.5)
ax_r.set_title("pixel_distance = polynomial(angle)\n(4 coefficients, fit at the factory)", fontsize=10.5, color=TEAL)
ax_r.tick_params(labelsize=8.5)

# generous clearance below the inset axes (their own xlabel/ticks extend
# past the strict plot_bottom_units boundary computed above)
cap_y = plot_bottom_units - 22
ax.text(50, cap_y,
        "This is exactly the 'intrinsic' part of the calibration file: 4 numbers\n"
        "(k1..k4) that describe this curve for that specific physical lens.",
        ha="center", fontsize=13, color="#333333")

ax.text(50, cap_y - 26,
        "A pinhole camera can only ever show you a bit less than 180° total\n"
        "(90° to each side) before the image would need infinite size. A fisheye\n"
        "lens is deliberately shaped (ground, multiple elements) so that light\n"
        "arriving from any angle, even nearly edge-on, still lands somewhere\n"
        "finite on the sensor — that's the whole trick that lets one camera\n"
        "cover almost half the world around the car.",
        ha="center", fontsize=11.5, color="#333333", linespacing=1.6)

footer(ax, "This is why a single flat 'homography' trick (normal panorama software) can't be used here.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 4: STEP 3
fig, ax = new_page()
header(ax, "STEP 3", "The Core Trick: Ask Backwards",
       "Instead of warping the photo, we ask each camera a question for every ground point")

cam_y = CONTENT_TOP - 22
grid_y0 = CONTENT_TOP - 74
gx = np.linspace(30, 70, 5)
gy = np.linspace(grid_y0, grid_y0 + 20, 3)
for x in gx:
    for y in gy:
        ax.add_patch(Circle((x, y), 0.8, color=TEAL, zorder=4))
ax.text(50, grid_y0 - 9, "a flat grid of real-world (X, Y) points laid on the ground around the car",
        ha="center", fontsize=11, color=TEAL)

camera_icon(ax, 50, cam_y, r=2.3, color=NAVY, label="fisheye camera")
for x in [35, 50, 65]:
    arrow(ax, x, grid_y0 + 20, 49, cam_y - 3, color=GREY, lw=1.3)
ax.text(50, cam_y + 13, "\"which of your pixels shows\nTHIS exact ground point?\"", ha="center",
        fontsize=12, color=NAVY, style="italic")

mid_y = (cam_y + grid_y0 + 20) / 2 - 4
box(ax, 5, mid_y, 24, 16, "grid point\n(X, Y) in meters", color=TEAL, fontsize=11)
arrow(ax, 22, mid_y, 34, grid_y0 + 24, color=TEAL)
box(ax, 71, mid_y, 24, 16, "camera answers:\npixel (u, v)", color=NAVY, fontsize=11)
arrow(ax, 65, cam_y + 2, 71, mid_y + 8, color=NAVY)

rule_y = grid_y0 - 18
ax.plot([4, DATA_W - 4], [rule_y, rule_y], color=LIGHT, lw=1.5, ls="--")
box(ax, 5, rule_y - 24, 43, 18, "WRONG WAY — Old way: warp the WHOLE\nphoto with one flat transform — breaks\non curved fisheye lines",
    color="#a8352b", fontsize=10.5)
box(ax, 52, rule_y - 24, 43, 18, "BETTER WAY — Our way: answer this\nquestion for every single grid point —\nworks for ANY lens shape",
    color=GREEN, fontsize=10.5)

footer(ax, "cv2.remap() then just fetches that pixel. Since all 4 cameras use the SAME grid, their answers are automatically aligned.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 5: STEP 4
fig, ax = new_page()
header(ax, "STEP 4", "Blending 4 Views Into 1 Picture",
       "Where two cameras' views overlap, fade smoothly between them instead of a hard cut")

rect_y = CONTENT_TOP - 38
rect_h = 30
for i, (x0, c, name) in enumerate([(8, BLUE, "Camera A"), (33, RED, "Camera B")]):
    ax.add_patch(Rectangle((x0, rect_y), 30, rect_h, facecolor=c, alpha=0.35, edgecolor=c, lw=2))
    ax.text(x0 + 15, rect_y + rect_h + 4, name, ha="center", fontsize=12, color=c, weight="bold")
ax.text(29, rect_y - 9, "overlap region\n(both cameras see this ground)", ha="center", fontsize=10.5, color="#333333")
arrow(ax, 29, rect_y - 2.5, 29, rect_y, color="#333333", lw=1.3)

box(ax, 6, rect_y - 45, 88, 24,
    "final pixel  =  (colorA × weightA + colorB × weightB) ÷ (weightA + weightB)",
    color=NAVY, fontsize=12.5)

plot_bottom_units = rect_y - 90
ax_w = plt.axes([0.12, plot_bottom_units / DATA_H, 0.32, 26 / DATA_H])
xw = np.linspace(0, 1, 100)
ax_w.plot(xw, np.clip(1 - xw * 1.6, 0, 1), color=BLUE, lw=2.5, label="weight A")
ax_w.plot(xw, np.clip(xw * 1.6 - 0.6, 0, 1), color=RED, lw=2.5, label="weight B")
ax_w.legend(fontsize=8.5, loc="center")
ax_w.set_title("each camera's confidence,\nleft to right", fontsize=10)
ax_w.set_xticks([]); ax_w.set_yticks([])

ax.text(66, plot_bottom_units + 13,
        "As you move left to right through the overlap,\n"
        "camera A's confidence fades out while camera\n"
        "B's fades in — like a photo dissolve, but driven\n"
        "by position instead of time.",
        ha="left", fontsize=10.5, color="#333333")

footer(ax, "This is basic feathering — every panorama tool does this. Getting the WEIGHTS right is where the real work is (see step 14).")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 6: STEP 5
fig, ax = new_page()
header(ax, "STEP 5", "Problem: The 4 Cameras Weren't Really Synced",
       "Same filename number ≠ same moment in time — we had to prove this, not assume it")

names = ["FV", "RV", "MVL", "MVR"]
top_line = CONTENT_TOP - 10
row_gap = 17
ycoords = [top_line - row_gap * i for i in range(4)]
colors2 = [BLUE, RED, TEAL, ORANGE]
for name, y, c in zip(names, ycoords, colors2):
    ax.plot([12, 92], [y, y], color="#cccccc", lw=1.5)
    ax.text(8, y, name, fontsize=12, color=c, weight="bold", ha="right", va="center")
    for tx in np.linspace(17, 87, 9):
        ax.plot([tx, tx], [y - 1.4, y + 1.4], color="#cccccc", lw=1)

fake_positions = {"FV": 20, "RV": 55, "MVL": 40, "MVR": 70}
for name, y, c in zip(names, ycoords, colors2):
    x = 17 + fake_positions[name] * 70 / 100
    ax.add_patch(Circle((x, y), 1.6, color=c, zorder=5))
    ax.text(x, y + 4.2, "\"00001\"", fontsize=9, ha="center", color=c)

match_x = 64
ax.plot([match_x, match_x], [ycoords[-1] - 3.5, ycoords[0] + 3.5], color=GREEN, lw=2, ls="--")
ax.text(match_x, ycoords[0] + 8, "one real timestamp,\nshared by all 4 cameras", fontsize=10, ha="center", color=GREEN)
for name, y, c in zip(names, ycoords, colors2):
    ax.add_patch(Circle((match_x, y), 1.3, facecolor="white", edgecolor=GREEN, lw=2, zorder=6))

ax.text(50, ycoords[-1] - 20, "Every camera's OWN frame counter — \"00001\" happens at a\ndifferent real time for each one!",
        ha="center", fontsize=12.5, color=RED, weight="bold")

box(ax, 10, ycoords[-1] - 58, 80, 20,
    "Fix: match frames by the hidden 'timestamp' field in\n"
    "vehicle_data/*.json instead of the filename number.\n\n"
    "The 4 timestamps that are EXACTLY equal (not just close)\n"
    "across all 4 cameras mark a genuinely synchronized moment.",
    color=NAVY, fontsize=11)
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 7: STEP 6a
fig, ax = new_page()
header(ax, "STEP 6a", "Problem: Tall Things Get Smeared",
       "Our whole trick assumes everything is lying flat on the ground — tall things break that")

ground_y = CONTENT_TOP - 66
cam_y = ground_y + 44
camera_icon(ax, 15, cam_y, r=2.1, color=NAVY, label="camera")
ax.plot([8, 92], [ground_y, ground_y], color="#999999", lw=2)
ax.text(10, ground_y - 5.5, "ground (Z=0)", fontsize=10, color="#999999", ha="left")

pole_top_y = ground_y + 25
ax.add_patch(Rectangle((49.5, ground_y), 1, pole_top_y - ground_y, facecolor=RED))
ax.add_patch(Circle((50, pole_top_y), 1.8, facecolor=RED))
ax.text(50, pole_top_y + 5, "a real sign / pole\n(has height)", ha="center", fontsize=11, color=RED)

arrow(ax, 15, cam_y, 50, pole_top_y - 1, color=NAVY, lw=1.6)
arrow(ax, 50, pole_top_y - 1, 78, ground_y, color=NAVY, lw=1.6, style="-")
ax.add_patch(Circle((78, ground_y), 1.5, facecolor=ORANGE, zorder=5))
ax.text(82, ground_y + 10, "algorithm assumes this\nray came from HERE\ninstead — wrong!",
        ha="left", fontsize=10.5, color=ORANGE)

ax.text(50, ground_y - 18, "Result: the sign's color gets painted far from where it really is —\n"
                           "the classic radial 'smear' you see on cars, poles, signs, curbs in every render.",
        ha="center", fontsize=12.5, color="#333333")

box(ax, 10, ground_y - 62, 80, 22,
    "Why this specific pattern: the algorithm only knows the DIRECTION of\n"
    "the ray from camera to pixel, not how far along it the real object sits.\n"
    "It resolves that ambiguity by assuming Z=0 (ground) every time — correct\n"
    "for the road, wrong for anything sticking up off it.",
    color=NAVY, fontsize=11)
footer(ax, "This can't be fixed with better calibration — it's a fundamental limit of the flat-ground assumption itself.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 8: STEP 6b
fig, ax = new_page()
header(ax, "STEP 6b", "Problem: Distant Ground Looks Blocky",
       "A second, unrelated cause of 'stretching' — measured directly, not guessed")

ground_y = CONTENT_TOP - 14
cam_y = ground_y + 7
camera_icon(ax, 15, cam_y, r=1.9)
ax.plot([8, 92], [ground_y, ground_y], color="#999999", lw=1.5)
for (gx_, label, ang_color) in [(30, "near\n(steep angle)", TEAL), (85, "far\n(grazing angle)", RED)]:
    arrow(ax, 15, cam_y, gx_, ground_y + 0.4, color=ang_color, lw=1.8)
    ax.add_patch(Circle((gx_, ground_y), 1.5, facecolor=ang_color, zorder=5))
    ax.text(gx_, ground_y - 6.5, label, ha="center", fontsize=10, color=ang_color)

plot_h_units = 40
plot_bottom_units = ground_y - 20 - plot_h_units
ax_bar = plt.axes([0.26, plot_bottom_units / DATA_H, 0.48, plot_h_units / DATA_H])
dist = [2, 4, 6, 8, 10]
ppm = [78, 25, 8, 3, 1.5]
ax_bar.bar([str(d) + "m" for d in dist], ppm, color=[TEAL, TEAL, ORANGE, RED, RED])
ax_bar.set_ylabel("real pixels available\nper meter of ground", fontsize=9.5)
ax_bar.set_title("measured on this dataset's front camera", fontsize=10.5, color=NAVY)
ax_bar.tick_params(labelsize=9)

cap_y = plot_bottom_units - 20
ax.text(50, cap_y,
        "At 2m: ~78 pixels per meter (plenty of detail). At 10m: only ~6.6\n"
        "(almost none). We were asking for a sharp 20m-wide picture from a\n"
        "camera that only has that much detail near itself.",
        ha="center", fontsize=12, color="#333333")
footer(ax, "Fix: shrink how far out we render (extent_m) to match what the camera can actually resolve.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 9: STEP 7
fig, ax = new_page()
header(ax, "STEP 7", "Problem: Cameras Disagree on Brightness",
       "Each camera auto-exposes independently — like 4 phones each picking their own brightness")

row_y = CONTENT_TOP - 24
for i, (x, c, val) in enumerate([(20, BLUE, "bright"), (45, RED, "dim"), (70, TEAL, "bright"), (90, ORANGE, "medium")]):
    ax.add_patch(Circle((x, row_y), 1.7, facecolor=c, edgecolor="white", lw=1.2, zorder=5))
    ax.add_patch(Circle((x, row_y), 1.7 * 0.45, facecolor="white", zorder=6))
    ax.add_patch(Rectangle((x - 4.5, row_y - 17), 9, 8, facecolor=c,
                            alpha={"bright": 0.9, "dim": 0.25, "medium": 0.55}[val]))
    ax.text(x, row_y - 21.5, val, ha="center", fontsize=10, color=c)

arrow(ax, 50, row_y - 30, 50, row_y - 39, color=NAVY, lw=2)
ax.text(50, row_y - 47,
        "Comparing brightness only in the small patch two cameras share didn't\n"
        "work — that patch mixes near and far ground seen at different angles, so\n"
        "different camera-pairs disagreed about which one was \"really\" too dark.",
        ha="center", fontsize=11.5, color="#333333")

box(ax, 8, row_y - 88, 84, 22,
    "Fix used instead: nudge each camera's OVERALL average\n"
    "brightness toward the group average. Cruder, but robust —\n"
    "and it worked.",
    color=GREEN, fontsize=12.5)
footer(ax, "Same idea as fixing 4 differently-exposed photos before making a collage.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 10: STEP 8
fig, ax = new_page()
header(ax, "STEP 8", "A Dead End, Investigated Properly",
       "The textbook fix for seam ghosting is feature matching — we tried it, and mostly ruled it out")

row_y = CONTENT_TOP - 28
box(ax, 5, row_y, 43, 28,
    "Attempt 1: match features on the\nflattened top-down image, warp\none camera to fit\n\n"
    "FAILED — Asphalt has almost no\ntexture to match → warp tore\nholes in the image",
    color="#a8352b", fontsize=11)
box(ax, 52, row_y, 43, 28,
    "Attempt 2: same idea, but\nmathematically constrained\n(rotation + shift only, no tearing)\n\n"
    "OK — Safe, but barely helped",
    color=ORANGE, fontsize=11)

arrow(ax, 73, row_y, 50, row_y - 20, color=NAVY, lw=2)
box(ax, 12, row_y - 56, 76, 26,
    "The REAL finding: fitted correction was TINY\n(< 0.3° rotation, < 20cm position)\n\n"
    "→ the calibration was already close to correct.\nThe leftover ghosting is mostly Step 6a (height),\nnot a calibration bug at all.",
    color=NAVY, fontsize=12)

ax.text(50, row_y - 68, "(the next page walks through exactly how each attempt worked)",
        ha="center", fontsize=10.5, color=GREY, style="italic")
footer(ax, "A cheap experiment that tells you \"this isn't the problem\" is just as valuable as one that fixes something.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 11: STEP 9 (NEW)
fig, ax = new_page()
header(ax, "STEP 9", "The Two Feature-Matching Techniques, In Detail",
       "What \"feature matching\" means here, and why the safer version behaves so differently")

col_top = CONTENT_TOP
left_x, right_x, col_w = 5, 52, 43

ax.text(left_x + col_w / 2, col_top, "Attempt 1: ORB + Homography",
        ha="center", fontsize=12.5, color="#a8352b", weight="bold")
ax.text(right_x + col_w / 2, col_top, "Attempt 2: Template Matching",
        ha="center", fontsize=12.5, color=GREEN, weight="bold")

diagram_y = col_top - 22
np.random.seed(7)
for dx, dy in [(0, 0), (14, 1), (6, -6), (20, -8), (28, 3)]:
    ax.add_patch(Circle((left_x + 6 + dx, diagram_y + dy), 0.9, facecolor="#a8352b", zorder=5))
    ax.add_patch(Circle((left_x + 6 + dx + 1.5, diagram_y + dy - 9), 0.9, facecolor="#a8352b", zorder=5))
    ax.plot([left_x + 6 + dx, left_x + 6 + dx + 1.5], [diagram_y + dy, diagram_y + dy - 9],
            color="#a8352b", lw=0.8, alpha=0.6)

gx0 = right_x + 4
for gxi in range(4):
    for gyi in range(3):
        cx = gx0 + gxi * 8
        cy = diagram_y + 2 - gyi * 6.5
        ax.add_patch(Rectangle((cx - 1.5, cy - 1.5), 3, 3, facecolor=GREEN, alpha=0.5))
        arrow(ax, cx, cy, cx + 2.2, cy + 1.3, color=GREEN, lw=0.9)

cap2_y = diagram_y - 22
ax.text(left_x + col_w / 2, cap2_y, "a few sparse keypoints (corners)\nmatched between two images —\nasphalt gives almost none of these",
        ha="center", fontsize=9.5, color="#333333")
ax.text(right_x + col_w / 2, cap2_y, "many small patches, each\nsearched for its best match nearby\n— works even on weak texture",
        ha="center", fontsize=9.5, color="#333333")

row2_y = cap2_y - 20
box(ax, left_x, row2_y, col_w, 15,
    "Correspondence: sparse ORB\nkeypoints + BFMatcher\n(Hamming distance)",
    color="#a8352b", fontsize=10)
box(ax, right_x, row2_y, col_w, 15,
    "Correspondence: dense normalized\ncross-correlation\n(cv2.matchTemplate)",
    color=GREEN, fontsize=10)

row3_y = row2_y - 19
box(ax, left_x, row3_y, col_w, 15,
    "Transform fit: full homography\n(RANSAC) — allows\nrotate + scale + SKEW",
    color="#a8352b", fontsize=10)
box(ax, right_x, row3_y, col_w, 15,
    "Transform fit: similarity only\n(RANSAC) — rotate + scale +\nshift, NO skew",
    color=GREEN, fontsize=10)

row4_y = row3_y - 19
box(ax, left_x, row4_y, col_w, 15,
    "Applied by warping the OUTPUT\nimage → can pull content\noff-canvas (holes)",
    color="#a8352b", fontsize=10)
box(ax, right_x, row4_y, col_w, 15,
    "Applied by shifting the INPUT\nquery grid → always samples\nvalidly (no holes)",
    color=GREEN, fontsize=10)

ax.text(50, row4_y - 14,
        "Same underlying idea (find matching ground points, use them to correct\n"
        "calibration) — but the SECOND version is deliberately too constrained\n"
        "to ever produce the tearing failure the first one did.",
        ha="center", fontsize=11.5, color="#333333")
footer(ax, "Both are legitimate classical computer-vision techniques — the difference is what each one is allowed to distort.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 12: STEP 10 (grain)
fig, ax = new_page()
header(ax, "STEP 10", "Cleaning Up Sensor Grain",
       "The grain was real (asphalt texture + sensor noise) — our own pipeline was magnifying it")

np.random.seed(3)
size = 20
base = np.zeros((size, size))
base[:, size // 2:] = 0.75
base[:, :size // 2] = 0.30
noise = (np.random.rand(size, size) - 0.5) * 0.30
noisy = np.clip(base + noise, 0, 1)
plain_blur = uniform_filter(noisy, size=5)
bilateral_like = np.hstack([
    uniform_filter(noisy[:, :size // 2], size=5),
    uniform_filter(noisy[:, size // 2:], size=5),
])

img_top = CONTENT_TOP - 8
img_h = 34
ax.text(50, img_top, "simulated noisy patch with a real brightness edge down the middle\n(dashed red line marks where the true edge is)",
        ha="center", fontsize=11.5, color="#333333")

panels = [(6, "before", noisy), (37, "plain blur\n(blurs the edge too!)", plain_blur),
          (68, "bilateral filter\n(edge stays sharp)", bilateral_like)]
img_bottom = img_top - 16 - img_h
for x0, title, grid in panels:
    ax.imshow(grid, extent=(x0, x0 + 26, img_bottom, img_bottom + img_h), cmap="gray", vmin=0, vmax=1)
    ax.plot([x0 + 13, x0 + 13], [img_bottom, img_bottom + img_h], color=RED, lw=1.6, ls=(0, (3, 2)))
    ax.text(x0 + 13, img_bottom - 5.5, title, ha="center", fontsize=10, color="#333333")

ax.text(50, img_bottom - 26,
        "Bilateral filter rule: only average nearby pixels that are BOTH close in\n"
        "position AND similar in color — so flat grain smooths out, but a real\n"
        "edge (very different color) stays sharp.",
        ha="center", fontsize=12, color="#333333")

box(ax, 10, img_bottom - 62, 80, 20,
    "Why apply it to the FINAL picture: a blur applied to the raw camera\n"
    "image before projection barely reaches the noisiest, closest-to-car\n"
    "areas — those get stretched over so many output pixels that a small\n"
    "input-side blur radius covers almost none of that stretched region.",
    color=NAVY, fontsize=10.5)
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 13: STEP 11 (self mask)
fig, ax = new_page()
header(ax, "STEP 11", "The Car Photographing Its Own Reflection",
       "That grey blob in every frame wasn't noise — it was the car seeing its own hood/door")

diagram_cy = CONTENT_TOP - 32
camera_icon(ax, 20, diagram_cy + 14, r=1.9, color=NAVY, label="front camera")
fov_wedge(ax, 20, diagram_cy + 14, 270, 100, 42, NAVY, alpha=0.10)
ax.add_patch(Polygon([(11, diagram_cy - 16), (29, diagram_cy - 16), (26, diagram_cy - 6), (14, diagram_cy - 6)],
                      facecolor="#555555"))
ax.text(20, diagram_cy - 21, "the car's own bumper,\nsticking into the shot", ha="center", fontsize=10, color="#555555")

box(ax, 50, diagram_cy + 6, 45, 24,
    "FAILED — Tried: find pixels that are\nIDENTICAL across many frames → only\ncaught the camera's own dead corners.\n\n"
    "Paint/chrome reflect the changing sky,\nso even the car's own body looks a\nbit different every frame.",
    color="#a8352b", fontsize=10.5)

box(ax, 50, diagram_cy - 26, 45, 28,
    "WORKED — AVERAGE many frames together.\n"
    "The changing world blurs into a flat haze,\nwhile anything truly fixed in position\n"
    "(even license-plate text!) stays crisp.\n\n"
    "Read that shape off once, hard-code it —\nexactly how real cars calibrate this.",
    color=GREEN, fontsize=10.5)

box(ax, 10, diagram_cy - 76, 80, 20,
    "One subtlety this revealed: \"fixed in position\" and \"fixed in\n"
    "appearance\" are different things. Position never changes for the\n"
    "car's own body — but appearance does, since it reflects the sky.",
    color=NAVY, fontsize=11)
footer(ax, "A one-time mask per camera, since the mount never moves relative to the lens.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 14: STEP 12 (bowl)
fig, ax = new_page()
header(ax, "STEP 12", "Fix: The 'Bowl' Trick Real Cars Use",
       "Instead of a flat ground, assume a shallow bowl — flat near the car, curving up at the edges")

base_y = CONTENT_TOP - 60
ax.plot([8, 45], [base_y, base_y], color=GREY, lw=2.5)
xs = np.linspace(45, 92, 60)
ys = base_y + ((xs - 45) / 47) ** 2 * 36
ax.plot(xs, ys, color=TEAL, lw=2.5)
ax.text(20, base_y - 6, "flat zone\n(near the car)", ha="center", fontsize=10.5, color="#555555")
ax.text(80, base_y + 42, "curves upward\n(far field)", ha="center", fontsize=10.5, color=TEAL, weight="bold")

cam_y = base_y + 13
camera_icon(ax, 8, cam_y, r=1.7, color=NAVY)
arrow(ax, 8, cam_y, 44, base_y + 0.3, color=NAVY, lw=1.3)
arrow(ax, 8, cam_y, 88, base_y + 36, color=NAVY, lw=1.3)
arrow(ax, 8, cam_y, 88, base_y, color=GREY, lw=1.2, style="->")
ax.text(50, base_y - 17, "old flat extension\n(grazing angle, low detail)", ha="center", fontsize=9.5, color=GREY)

ax.text(50, base_y - 34,
        "Two wins at once: distant blur gets curved out of the main view, AND the\n"
        "ray now hits at a less grazing angle — directly helping the resolution\n"
        "problem from the last page too.",
        ha="center", fontsize=12, color="#333333")

box(ax, 10, base_y - 68, 80, 18,
    "The car's own hood/roofline never crosses the flat zone in practice —\n"
    "which is exactly why the mask from the last page and the bowl from\n"
    "this page can be calibrated independently of each other.",
    color=NAVY, fontsize=10.5)
footer(ax, "This is literally what production '360 camera' displays in real cars use — not because it's geometrically correct, but because it looks right.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 15: STEP 13 (depth)
fig, ax = new_page()
header(ax, "STEP 13", "Trying Real Depth Estimation",
       "The 'correct' fix for Step 6a — tried, and failed in a very instructive way")

img_top = CONTENT_TOP - 8
img_h = 36
plot_bottom_units = img_top - img_h
ax_a = plt.axes([0.09, plot_bottom_units / DATA_H, 0.38, img_h / DATA_H])
ax_a.imshow(np.tile(np.linspace(0, 1, 100), (100, 1)).T, cmap="viridis")
ax_a.set_title("what the depth model learned\nfrom ordinary photos:\n\"top = far, bottom = close\"", fontsize=9.5)
ax_a.axis("off")

ax_b = plt.axes([0.55, plot_bottom_units / DATA_H, 0.38, img_h / DATA_H])
yy, xx = np.mgrid[-1:1:100j, -1:1:100j]
rr = np.sqrt(xx**2 + yy**2)
ax_b.imshow(rr, cmap="viridis")
ax_b.set_title("what's ACTUALLY true for a\n180° fisheye photo:\ndepth varies in RINGS from the center", fontsize=9.5)
ax_b.axis("off")

cap_y = plot_bottom_units - 16
ax.text(50, cap_y, "Feeding a fisheye image into a model that assumes the left pattern\n"
                   "produces a badly wrong depth map — which we then measured directly:\n"
                   "it came out as an almost featureless top-to-bottom gradient.",
        ha="center", va="center", fontsize=11.5, color="#333333")
box(ax, 10, cap_y - 42, 80, 22,
    "Result when we tried to use it: everything collapsed into ring/donut\n"
    "shapes instead of a real scene. Root tool exists (OmniDet, trained on\n"
    "fisheye video specifically) — just not integrated this session.",
    color=NAVY, fontsize=11)
footer(ax, "A pretrained model is only as good as how similar its training photos were to what you feed it.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 16: STEP 14 (seams)
fig, ax = new_page()
header(ax, "STEP 14", "Fixing Double Images at the Seams",
       "The last fix: change WHEN a camera is allowed to feel \"fully confident\"")

rect_top = CONTENT_TOP - 10
rect_h = 34
for i, (x0, title) in enumerate([(6, "OLD: fades only near the very edge"),
                                  (52, "NEW: fades based on distance to\nEACH CAMERA'S OWN center")]):
    ax.text(x0 + 20, rect_top, title, ha="center", fontsize=10.5, color=NAVY, weight="bold")
    ax.add_patch(Rectangle((x0, rect_top - rect_h - 4), 24, rect_h, facecolor=BLUE, alpha=0.30))
    ax.add_patch(Rectangle((x0 + 14, rect_top - rect_h - 4), 24, rect_h, facecolor=RED, alpha=0.30))
    if i == 0:
        ax.add_patch(Rectangle((x0 + 14, rect_top - rect_h - 4), 10, rect_h, facecolor=PURPLE, alpha=0.55))
        ax.text(x0 + 19, rect_top - rect_h - 11, "wide 50/50 zone\n= visible double image", ha="center", fontsize=9.5, color=PURPLE)
    else:
        ax.add_patch(Rectangle((x0 + 22, rect_top - rect_h - 4), 2, rect_h, facecolor=PURPLE, alpha=0.85))
        ax.text(x0 + 23, rect_top - rect_h - 11, "narrow crossover line\n= one clean version wins", ha="center", fontsize=9.5, color=PURPLE)

cap_y = rect_top - rect_h - 28
ax.text(50, cap_y,
        "Overlap zones are HUGE (measured: 20-52% of the picture) — so with the\n"
        "old method, both cameras stayed \"fully confident\" across most of that\n"
        "shared area, blending two different (parallax-shifted) views of the same object.",
        ha="center", fontsize=12, color="#333333")

box(ax, 10, cap_y - 40, 80, 20,
    "The distance-transform trick: each camera's confidence keeps growing\n"
    "the deeper you are into its own footage (not just near its own edge),\n"
    "so whichever camera is more central at a point dominates almost fully.",
    color=NAVY, fontsize=11)
footer(ax, "Doesn't fix the underlying height-parallax — but now you see ONE clean (if imperfect) version instead of two overlapping ghosts.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 17: SUMMARY
fig, ax = new_page()
header(ax, "SUMMARY", "The Full Pipeline, Start to Finish")

stages = [
    "4 raw fisheye images\n+ calibration files",
    "match frames by\nreal timestamp",
    "project ground/bowl\ngrid into each camera",
    "match brightness\nacross cameras",
    "blend overlaps\n(distance-based)",
    "mask out the\ncar's own body",
    "smooth grain\n(bilateral filter)",
    "final surround view",
]
col_w, col_h, col_gap_x, row_gap_y = 40, 20, 8, 8
left_x, right_x = 8, left_x + col_w + col_gap_x
n_rows = 4
grid_top = CONTENT_TOP - 2
for i, s in enumerate(stages):
    row = i // 2
    is_left = (i % 2 == 0)
    x = left_x if is_left else right_x
    y = grid_top - row * (col_h + row_gap_y) - col_h
    c = NAVY if i in (0, len(stages) - 1) else BLUE
    box(ax, x, y, col_w, col_h, s, color=c, fontsize=10)

    if i < len(stages) - 1:
        if is_left:
            # arrow across to the right box in the same row
            arrow(ax, x + col_w, y + col_h / 2, right_x, y + col_h / 2, color=GREY, lw=1.8)
        else:
            # arrow down+left to the start of the next row
            next_y = grid_top - (row + 1) * (col_h + row_gap_y) - col_h
            arrow(ax, x + col_w / 2, y, left_x + col_w / 2, next_y + col_h, color=GREY, lw=1.8)

summary_y = grid_top - n_rows * (col_h + row_gap_y) - 12
ax.text(50, summary_y,
        "Almost every fix above came from MEASURING the actual failure (pixel\n"
        "densities, brightness values, timestamp overlaps, fitted correction sizes)\n"
        "rather than guessing from how a picture looked. The times we skipped that\n"
        "(feature-matching on asphalt, a generic depth model) — we found out\n"
        "quickly and cheaply that it didn't work.",
        ha="center", fontsize=12, color="#333333", linespacing=1.7)
footer(ax, "End of walkthrough.")
PP.savefig(fig)
plt.close(fig)

PP.close()
print("Saved", REPORT_PATH)
