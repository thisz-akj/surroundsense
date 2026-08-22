"""
Performance report comparing 5 correspondence-finding techniques for the
seam-alignment correction step, all measured on the same 10-frame sandbox
with everything else in the pipeline (bowl surface, gain correction,
self-mask, denoise) held constant.
"""

import json
import os
import pickle
import statistics
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import EXPERIMENTS_DIR, DOCS_DIR  # noqa: E402

REPORT_PATH = os.path.join(DOCS_DIR, "feature_matching_performance_report.pdf")

PAGE_SIZE = (8.27, 11.69)  # A4 portrait
DATA_W = 100.0
DATA_H = DATA_W * PAGE_SIZE[1] / PAGE_SIZE[0]
PT_PER_UNIT_X = PAGE_SIZE[0] * 72 / DATA_W

NAVY = "#1c2b4a"
GREY = "#8a8f98"
LIGHT_GRID = "#e4e6ea"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GOOD = "#1baf7a"
BAD = "#e34948"

# Fixed categorical order (validated adjacent-pair palette, dataviz skill default)
TECHNIQUES = ["template_matching", "sift", "ecc", "phase_correlation", "loftr"]
LABELS = {
    "template_matching": "Template\nmatching",
    "sift": "SIFT",
    "ecc": "ECC\n(tiled)",
    "phase_correlation": "Phase\ncorrelation",
    "loftr": "LoFTR",
}
COLORS = {
    "template_matching": "#2a78d6",  # blue
    "sift": "#eb6834",               # orange
    "ecc": "#1baf7a",                # aqua
    "phase_correlation": "#eda100",  # yellow
    "loftr": "#e87ba4",              # magenta
}
PAIRS = ["FV-MVL", "FV-MVR", "RV-MVL", "RV-MVR"]

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


def draw_paragraph(ax, x, y_top, text, width_units, fontsize=10.5, color=TEXT,
                    char_w=0.50, line_spacing=1.45, weight="normal", ha="left"):
    """
    Wraps `text` to fit `width_units` at `fontsize` (computing chars-per-line
    from the same physical-size assumption _fit_fontsize uses), draws it
    anchored at its top-left, and returns the height (in data units) it
    consumed -- so callers can advance y by exactly that much instead of a
    guessed fixed offset (which silently overflows for long paragraphs).
    """
    avail_pt = width_units * PT_PER_UNIT_X * 0.97
    chars_per_line = max(int(avail_pt / (fontsize * char_w)), 10)
    wrapped = textwrap.fill(text, width=chars_per_line)
    n_lines = wrapped.count("\n") + 1
    line_height = fontsize * line_spacing / PT_PER_UNIT_X
    ax.text(x, y_top, wrapped, fontsize=fontsize, color=color, va="top", ha=ha,
            weight=weight, linespacing=line_spacing)
    return n_lines * line_height


HEADER_STEP_Y = DATA_H - 7
HEADER_TITLE_Y = DATA_H - 16
HEADER_SUB_Y = DATA_H - 23.5
HEADER_RULE_Y = DATA_H - 27.5
CONTENT_TOP = DATA_H - 34
FOOTER_Y = 5


def header(ax, tag, title, subtitle=None):
    ax.text(4, HEADER_STEP_Y, tag, fontsize=12.5, color="#e07b39", weight="bold")
    fs = _fit_fontsize(title, DATA_W - 8, 19)
    ax.text(4, HEADER_TITLE_Y, title, fontsize=fs, color=NAVY, weight="bold")
    if subtitle:
        fs_sub = _fit_fontsize(subtitle, DATA_W - 8, 11.5, char_w=0.52)
        ax.text(4, HEADER_SUB_Y, subtitle, fontsize=fs_sub, color=TEXT_SECONDARY, style="italic")
    ax.plot([4, DATA_W - 4], [HEADER_RULE_Y, HEADER_RULE_Y], color=LIGHT_GRID, lw=2)


def footer(ax, note):
    fs = _fit_fontsize(note, DATA_W - 16, 9, char_w=0.50)
    ax.text(DATA_W / 2, FOOTER_Y, note, fontsize=fs, color=GREY, ha="center", style="italic")


def box(ax, x, y, w, h, text, color=NAVY, fontsize=11, text_color="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                        linewidth=0, facecolor=color)
    ax.add_patch(b)
    fs = _fit_fontsize(text, w, fontsize)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=text_color)


_RAW = json.load(open(os.path.join(EXPERIMENTS_DIR, "matching_shootout", "raw_results.json")))
_TIMING_SECONDS = {"template_matching": 29, "sift": 31, "ecc": 33, "phase_correlation": 25, "loftr": 237}

rows = {}
for t in TECHNIQUES:
    before, after, ncorr = _RAW[t]["before"], _RAW[t]["after"], _RAW[t]["n_corr"]
    mvl = sum(n.get("MVL", 0) for n in ncorr) / len(ncorr)
    mvr = sum(n.get("MVR", 0) for n in ncorr) / len(ncorr)
    pcts = {}
    for p in PAIRS:
        b_vals = [b[p] for b in before if b.get(p) is not None]
        a_vals = [a[p] for a in after if a.get(p) is not None]
        b_avg, a_avg = sum(b_vals) / len(b_vals), sum(a_vals) / len(a_vals)
        pcts[p] = 100 * (b_avg - a_avg) / b_avg
    avg_pct = sum(pcts.values()) / len(pcts)
    rows[t] = {"mvl": mvl, "mvr": mvr, "pcts": pcts, "avg_pct": avg_pct,
               "time": _TIMING_SECONDS[t], "std": statistics.pstdev(list(pcts.values()))}

PP = PdfPages(REPORT_PATH)


# ============================================================ PAGE 1: TITLE + METHODOLOGY
fig, ax = new_page()
ax.text(50, DATA_H - 18, "Feature-Matching Technique", fontsize=22, color=NAVY, weight="bold", ha="center")
ax.text(50, DATA_H - 25.5, "Performance Report", fontsize=22, color=NAVY, weight="bold", ha="center")
ax.text(50, DATA_H - 32.5, "5 correspondence-finding techniques, 1 pipeline, measured head-to-head",
        fontsize=12, color=TEXT_SECONDARY, ha="center", style="italic")

y = DATA_H - 46
ax.text(6, y, "Held constant across every run", fontsize=13, color=NAVY, weight="bold")
y -= 7
for line in [
    "Bowl-shaped projection surface, gain correction, ego-vehicle self-mask, "
    "bilateral-filter denoising",
    "The same RANSAC similarity-transform fit (rotation + scale + shift only, no "
    "skew), applied to the query grid — identical downstream logic for every technique",
]:
    ax.text(9, y, "•", fontsize=10.8, color=TEXT, va="top")
    h = draw_paragraph(ax, 12.5, y, line, 84, fontsize=10.8, char_w=0.51)
    y -= h + 4

y -= 6
ax.text(6, y, "What varied — the correspondence-finding technique", fontsize=13, color=NAVY, weight="bold")
y -= 7.5
for name in TECHNIQUES:
    ax.add_patch(plt.Rectangle((9, y - 1.6), 3, 3, facecolor=COLORS[name]))
    label = LABELS[name].replace("\n", " ")
    ax.text(14, y, label, fontsize=11, color=TEXT, va="center")
    y -= 6.2

y -= 6
ax.text(6, y, "Test set & metric", fontsize=13, color=NAVY, weight="bold")
y -= 7
for line in [
    "10 sandbox frames × 4 camera-overlap pairs each (FV-MVL, FV-MVR, RV-MVL, RV-MVR)",
    "Metric: seam-disagreement score (mean pixel difference where two cameras render "
    "the same ground) — reported as % improvement from before to after correction, "
    "lower score is better, averaged over the 10 frames",
]:
    ax.text(9, y, "•", fontsize=10.8, color=TEXT, va="top")
    h = draw_paragraph(ax, 12.5, y, line, 84, fontsize=10.8, char_w=0.51)
    y -= h + 4

footer(ax, "Full methodology for the underlying pipeline: see the companion beginner PDF (surround_view_explainer.pdf).")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 2: CORRESPONDENCE COUNTS
fig, ax = new_page()
header(ax, "RESULT 1", "How Many Correspondences Did Each Find?",
       "Averaged across all 10 frames — the raw amount of signal each technique had to work with")

plot_h = 68
bottom_frac = (CONTENT_TOP - 8 - plot_h) / DATA_H
axp = plt.axes([0.13, bottom_frac, 0.80, plot_h / DATA_H])

x_base = np.arange(5)
bar_w = 0.32
mvl_vals = [rows[t]["mvl"] for t in TECHNIQUES]
mvr_vals = [rows[t]["mvr"] for t in TECHNIQUES]
bars_mvl = axp.bar(x_base - bar_w / 2, mvl_vals, width=bar_w * 0.94,
                    color=[COLORS[t] for t in TECHNIQUES], zorder=3)
bars_mvr = axp.bar(x_base + bar_w / 2, mvr_vals, width=bar_w * 0.94,
                    color=[COLORS[t] for t in TECHNIQUES], zorder=3, hatch="////",
                    edgecolor="white", linewidth=0.6)
axp.set_xticks(x_base)
axp.set_xticklabels([LABELS[t] for t in TECHNIQUES], fontsize=9)
axp.set_ylabel("avg. correspondences found per frame", fontsize=9.5)
axp.grid(axis="y", color=LIGHT_GRID, lw=1, zorder=0)
axp.set_axisbelow(True)
axp.spines[["top", "right"]].set_visible(False)
axp.tick_params(axis="y", labelsize=9)
for bars, vals in [(bars_mvl, mvl_vals), (bars_mvr, mvr_vals)]:
    for b, v in zip(bars, vals):
        axp.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8.3, color=TEXT, weight="bold")

# custom two-entry legend for solid=MVL / hatched=MVR (color already means technique)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor="#999999", label="Movable camera: MVL"),
                  Patch(facecolor="#999999", hatch="////", edgecolor="white", label="Movable camera: MVR")]
axp.legend(handles=legend_handles, fontsize=8.5, loc="upper left", frameon=False)

cap_y = bottom_frac * DATA_H - 14
ax.text(50, cap_y,
        "LoFTR finds 4-6x more correspondences than every other technique — the\n"
        "expected result, since it's specifically built for exactly this failure mode\n"
        "(low-texture, repetitive patterns like asphalt) that starves classical detectors.",
        ha="center", fontsize=11.5, color=TEXT)
footer(ax, "ECC and phase correlation are tiled (many small patches); counted here is how many tiles passed their confidence threshold.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 3: SEAM IMPROVEMENT
fig, ax = new_page()
header(ax, "RESULT 2", "Did It Actually Reduce Ghosting?",
       "% change in seam-disagreement score per camera pair — positive = less ghosting, negative = worse")

plot_h = 78
bottom_frac = (CONTENT_TOP - 10 - plot_h) / DATA_H
axp = plt.axes([0.11, bottom_frac, 0.85, plot_h / DATA_H])

n_pairs, n_tech = len(PAIRS), len(TECHNIQUES)
group_w = 0.8
bar_w = group_w / n_tech
x_base = np.arange(n_pairs)

for i, t in enumerate(TECHNIQUES):
    xs = x_base - group_w / 2 + bar_w * (i + 0.5)
    vals = [rows[t]["pcts"][p] for p in PAIRS]
    bars = axp.bar(xs, vals, width=bar_w * 0.92, color=COLORS[t], label=LABELS[t].replace("\n", " "), zorder=3)
    for b, v in zip(bars, vals):
        axp.annotate(f"{v:+.1f}", (b.get_x() + b.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 3 if v >= 0 else -11),
                     ha="center", fontsize=6.6, color=TEXT)

axp.axhline(0, color="#444444", lw=1.2, zorder=2)
axp.set_xticks(x_base)
axp.set_xticklabels(PAIRS, fontsize=10)
axp.set_ylabel("% change in seam-disagreement score\n(higher = more improvement)", fontsize=9.5)
axp.grid(axis="y", color=LIGHT_GRID, lw=1, zorder=0)
axp.set_axisbelow(True)
axp.spines[["top", "right"]].set_visible(False)
axp.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
axp.tick_params(axis="y", labelsize=9)

cap_y = bottom_frac * DATA_H - 16
ax.text(50, cap_y,
        "LoFTR is the only technique with a consistent positive trend (3 of 4 pairs\n"
        "improved, up to +6.8%). SIFT's average looks similar to baseline, but that\n"
        "hides high variance — it helps 2 pairs and hurts 2, swinging ±9%.",
        ha="center", fontsize=11.3, color=TEXT)
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 4: TIMING + RANKING TABLE
fig, ax = new_page()
header(ax, "RESULT 3", "Timing, and the Overall Ranking",
       "10 frames, 4 camera-pairs each — everything else identical")

plot_h = 34
bottom_frac = (CONTENT_TOP - 6 - plot_h) / DATA_H
axp = plt.axes([0.16, bottom_frac, 0.76, plot_h / DATA_H])
vals = [rows[t]["time"] for t in TECHNIQUES]
bars = axp.bar(range(5), vals, color=[COLORS[t] for t in TECHNIQUES], width=0.6, zorder=3)
axp.set_xticks(range(5))
axp.set_xticklabels([LABELS[t] for t in TECHNIQUES], fontsize=9)
axp.set_ylabel("seconds for 10 frames\n(4 camera-pairs each)", fontsize=9.5)
axp.grid(axis="y", color=LIGHT_GRID, lw=1, zorder=0)
axp.set_axisbelow(True)
axp.spines[["top", "right"]].set_visible(False)
for b, v in zip(bars, vals):
    axp.annotate(f"{v}s", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                 xytext=(0, 3), ha="center", fontsize=9.5, color=TEXT, weight="bold")
axp.tick_params(axis="y", labelsize=9)

table_top = bottom_frac * DATA_H - 9
ax.text(50, table_top, "Overall ranking (avg improvement across all 4 pairs)", fontsize=12.5,
        color=NAVY, weight="bold", ha="center")

ranked = sorted(TECHNIQUES, key=lambda t: -rows[t]["avg_pct"])
col_x = [8, 42, 60, 78]
headers = ["Technique", "Avg %", "Std dev", "Verdict"]
row_y = table_top - 7
for cx, h in zip(col_x, headers):
    ax.text(cx, row_y, h, fontsize=10, color=NAVY, weight="bold")
ax.plot([6, 94], [row_y - 2, row_y - 2], color=LIGHT_GRID, lw=1.5)
row_y -= 6.5

verdicts = {
    "loftr": "Best — real, consistent gain",
    "template_matching": "Baseline — no real effect",
    "phase_correlation": "No real effect",
    "sift": "Unstable — cancels out",
    "ecc": "Net negative",
}
for t in ranked:
    r = rows[t]
    ax.add_patch(plt.Rectangle((6, row_y - 3.2), 3, 3, facecolor=COLORS[t]))
    ax.text(col_x[0] + 4, row_y, LABELS[t].replace("\n", " "), fontsize=10, color=TEXT, va="center")
    ax.text(col_x[1], row_y, f"{r['avg_pct']:+.2f}%", fontsize=10, color=TEXT, va="center")
    ax.text(col_x[2], row_y, f"{r['std']:.2f}", fontsize=10, color=TEXT, va="center")
    fs = _fit_fontsize(verdicts[t], 24, 9.5, char_w=0.55)
    ax.text(col_x[3], row_y, verdicts[t], fontsize=fs, color=TEXT, va="center")
    row_y -= 6.5

footer(ax, "LoFTR runs on CPU here (no GPU available) — the 237s/10-frames cost would shrink substantially on a GPU.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 5: PER-TECHNIQUE ANALYSIS
fig, ax = new_page()
header(ax, "ANALYSIS", "What Happened With Each Technique, And Why")

analysis = [
    ("loftr", GOOD, "Winner. Its correspondence count (300/198) dwarfs everything else because it's a "
                    "detector-free deep model trained specifically to find matches in low-texture, repetitive "
                    "regions — exactly what asphalt is. More (good) correspondences means the RANSAC similarity "
                    "fit has far more to work with, producing a more stable, better-conditioned correction. "
                    "Visually confirmed on a railing spanning the RV-MVL seam: noticeably smoother, less kinked "
                    "than the baseline. Cost: ~8x slower than the others on CPU."),
    ("sift", "#eb6834", "A real step up in match quality over ORB (which found almost nothing on this asphalt "
                        "in the very first attempt) — SIFT found a comparable count to the existing NCC method. "
                        "But the average hides the real story: results swing from +8.7% to -8.9% across "
                        "different camera pairs. Classical sparse detectors on repetitive texture (aggregate "
                        "grain, lane-paint patterns) can produce confidently wrong matches, not just too few "
                        "of them — RANSAC can't always filter that out."),
    ("ecc", "#1baf7a", "Failed cleanly, and instructively. Tried first on the whole overlap region: didn't "
                       "converge at all (too much of the region is parallax-affected content, not matching "
                       "ground). Tiled down to 90x90px patches: most tiles still failed to converge, and the "
                       "few that did had low confidence. Whole/large-region pixel correlation is a poor fit "
                       "when the region mixes real matching ground with parallax-shifted objects."),
    ("phase_correlation", "#eda100", "Same root problem as ECC, different mechanism. FFT-based phase "
                          "correlation assumes the two patches are nearly identical up to a pure shift; "
                          "measured confidence scores here topped out around 0.2 (out of 1.0) even after "
                          "tiling, versus the ≥0.45 threshold needed to trust a match. Essentially no usable "
                          "signal from this technique on this data."),
    ("template_matching", "#2a78d6", "The existing method (dense NCC, many small patches). Consistently "
                          "near-zero effect (±2%) — which is itself the key earlier finding: the actual "
                          "calibration error here is tiny (<0.3° rotation, <20cm position), so there just "
                          "isn't much for ANY calibration-correction technique to fix. Most of the visible "
                          "'ghosting' is height-parallax, not a calibration bug."),
]

y = CONTENT_TOP
for name, color, text in analysis:
    ax.add_patch(plt.Rectangle((4, y - 3.2), 4, 4, facecolor=color))
    ax.text(10, y - 1.2, LABELS[name].replace("\n", " "), fontsize=12.5, color=NAVY, weight="bold")
    y -= 8
    h = draw_paragraph(ax, 6, y, text, 90, fontsize=9.6, char_w=0.50, line_spacing=1.5)
    y -= h + 6

footer(ax, "\"Verdict\" in the ranking table reflects net effect across the 4 pairs, not raw correspondence count.")
PP.savefig(fig)
plt.close(fig)


# ============================================================ PAGE 6: RECOMMENDATION
fig, ax = new_page()
header(ax, "RECOMMENDATION", "What To Actually Do With This")

box(ax, 6, CONTENT_TOP - 26, 88, 26,
    "If you want to swap the seam-alignment technique for the full 542-frame\n"
    "batch: use LoFTR. It's the only technique that produced a real, consistent\n"
    "improvement (up to +6.8% on individual pairs, +2.88% average) rather than\n"
    "noise-level change or an unstable swing between helping and hurting.",
    color=NAVY, fontsize=12.5)

y = CONTENT_TOP - 38
ax.text(6, y, "But weigh the cost:", fontsize=13, color=NAVY, weight="bold")
y -= 7
for line in [
    "~8x slower than every other technique tested (237s vs 25-33s for 10 frames on CPU)",
    "At that rate, the full 542-frame batch would take roughly 3-4 hours on this machine",
    "Requires the PyTorch + kornia dependency (already installed in .venv, ~1.4GB)",
    "The gain, while real, is modest in absolute terms (a few percent) — it does not touch "
    "the height-parallax smearing, which remains the dominant visible artifact",
]:
    ax.text(9, y, "•", fontsize=11, color=TEXT, va="top")
    h = draw_paragraph(ax, 12.5, y, line, 82, fontsize=11, char_w=0.52)
    y -= h + 2.8

y -= 5
box(ax, 6, y - 30, 88, 30,
    "Practical suggestion: given the modest absolute gain and the ~8x runtime cost,\n"
    "LoFTR is worth using if geometric seam quality specifically matters for your next\n"
    "step (e.g. training data curation). If you're optimizing for throughput instead,\n"
    "the existing dense-NCC template matching remains a reasonable default — it costs\n"
    "almost nothing extra and the seam quality it leaves behind is already close to\n"
    "calibration-optimal, per the earlier finding that the true miscalibration is tiny.",
    color=NAVY, fontsize=11.5)

footer(ax, "End of report.")
PP.savefig(fig)
plt.close(fig)

PP.close()
print("Saved", REPORT_PATH)
