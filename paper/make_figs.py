#!/usr/bin/env python3
"""
Figures for "When a Blank Is Not a Control".

Every number is transcribed from the filed result artefacts (wia_dprime.json,
wia_baseline.json, wia_mrate.json, wia_xklen_cross_20554.json, wia_bench2, the
PAI lane).  This script only draws; it computes nothing reported as a result.

House style, identical in every figure:
  * Times New Roman for all text, including mathematics (custom mathtext set),
    TrueType-embedded (pdf.fonttype 42) so no Type 3 fonts reach the PDF.
  * Sentence case throughout; panel labels are a bold "(a)"/"(b)" followed by a
    sentence-case title, left-aligned.
  * One type scale: 8.5 pt for labels, ticks and legends, 9 pt for panel titles,
    7.8 pt for value annotations.  One set of line weights: 0.6 pt spines,
    0.45 pt grid, 1.1 pt data strokes, 0.8 pt reference lines.
  * Every figure is drawn at the width it occupies on the page and included at
    natural size, so the point sizes above are the sizes a reader sees.

Colour semantics: blue/orange means sighted/blind and nothing else (validated
colourblind-safe, protan dE 23.3, normal dE 30.3).  Every other contrast (two
architectures, two readouts, the mismatched image) is drawn in neutral greys or
black, so the paper uses one palette.  Every pairing also carries a non-colour cue (hatch, marker
shape, row label or block rule) so the figures survive greyscale print.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

SIGHTED = "#1F5FA9"
BLIND   = "#C4620A"
INK     = "#1a1a1a"
MUTED   = "#5f5f5f"
GRID    = "#dcdcd8"
SURFACE = "#ffffff"
PALE_S  = "#dbe6f3"
PALE_B  = "#f6e3d2"
BAND    = "#ebebe8"
ARCH_A  = "#333333"      # architecture 1 (fig_robustness): dark neutral, circles
ARCH_B  = "#9a9a9a"      # architecture 2: light neutral, squares
READ_A  = "#4a4a4a"      # free generation (fig_forced)
READ_B  = "#bdbdbd"      # forced choice

FS   = 8.5      # labels, ticks, legends
FS_T = 9.0      # panel titles
FS_S = 7.8      # value annotations
LW_AX, LW_GRID, LW_DATA, LW_REF, LW_EDGE = 0.6, 0.45, 1.1, 0.8, 0.8
MS = 5.0        # marker size
TW = 5.5        # text width, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.fallback": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS_T,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.titlepad": 5, "axes.labelpad": 3,
    "axes.edgecolor": MUTED, "axes.linewidth": LW_AX,
    "xtick.major.width": LW_AX, "ytick.major.width": LW_AX,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "lines.linewidth": LW_DATA, "patch.linewidth": LW_EDGE,
    "hatch.linewidth": 0.7,
    "legend.frameon": False, "legend.handlelength": 1.3,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.3,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.edgecolor": "none",
})


def sg(v, nd=2):
    """Signed number with a typographic minus."""
    return f"{v:+.{nd}f}".replace("-", "\u2212")


def recessive(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=LW_GRID, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ptitle(ax, letter, text):
    """Bold panel label, then a sentence-case title, left-aligned."""
    ax.set_title(rf"$\mathbf{{({letter})}}$  {text}", loc="left")


def stitle(ax, text):
    ax.set_title(text, loc="left")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", pad_inches=0.015,
                facecolor="white", edgecolor="none")
    plt.close(fig)


def fit_box(ax, texts, pad_x, pad_y, **kw):
    """Draw a rectangle around the union of the given Text objects (data units)."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bbs = [t.get_window_extent(renderer=r) for t in texts]
    x0 = min(b.x0 for b in bbs); x1 = max(b.x1 for b in bbs)
    y0 = min(b.y0 for b in bbs); y1 = max(b.y1 for b in bbs)
    (dx0, dy0), (dx1, dy1) = inv.transform([(x0, y0), (x1, y1)])
    ax.add_patch(Rectangle((dx0 - pad_x, dy0 - pad_y), dx1 - dx0 + 2 * pad_x,
                           dy1 - dy0 + 2 * pad_y, zorder=2, **kw))
    return dx0 - pad_x, dy0 - pad_y, dx1 + pad_x, dy1 + pad_y


def inch_canvas(w, h):
    """A figure whose single axes spans it, with data units equal to inches."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


# ============================================================ fig_lead
# Figure 1.  Panel (a) doubles as the positive control: the image raises J by about +0.2
# WITHIN each condition (wia_dprime.json, lanes/F1/ci/sight_gap_cap{1,5}), and by the same
# amount in both, which is why it contributes nothing to the one->five contrast under grey.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(TW, 1.98),
                               gridspec_kw={"width_ratios": [1.0, 1.0]})
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.86, wspace=0.36)
BW = 0.36
x = np.arange(2)
g_s = [0.18814, 0.67401]
g_b = [-0.02191, 0.47997]
axL.bar(x - BW/2, g_s, BW, color=SIGHTED, label=r"Sighted (rise $+0.49$)",
        zorder=3, edgecolor=SURFACE, linewidth=LW_EDGE)
axL.bar(x + BW/2, g_b, BW, color=BLIND, label=r"Grey field (rise $+0.50$)",
        zorder=3, edgecolor=SURFACE, linewidth=LW_EDGE, hatch="///")
axL.axhline(0, color=MUTED, linewidth=LW_REF, zorder=2)
for xi, v in list(zip(x - BW/2, g_s)) + list(zip(x + BW/2, g_b)):
    axL.text(xi, v + (0.03 if v >= 0 else -0.035), sg(v), ha="center",
             va="bottom" if v >= 0 else "top", fontsize=FS_S, color=INK)
axL.set_xticks(x); axL.set_xticklabels(["One scene", "Five scenes"])
axL.set_ylabel(r"$J = H - F$")
axL.set_ylim(-0.13, 1.22); axL.set_yticks([0, 0.5, 1.0])
axL.set_xlim(-0.55, 1.55)
ptitle(axL, "a", "The score with and without the image")
axL.legend(loc="upper left")
recessive(axL)

# (b) The 2x2 on one population: {grey, mismatched image} x {all cells, cells whose sighted
# continuations both end before the budget}, 3,500 images (cpw_2x2_extra.json; ratio =
# ablated rise / sighted rise, 95% percentile intervals, B = 4000 clustered on the image).
MM = INK
reg = np.array([0.0, 1.0])
r_g = [1.0066, 0.7099]; lo_g = [0.9612, 0.5555]; hi_g = [1.0545, 0.8688]
r_m = [0.6754, 0.4741]; lo_m = [0.6350, 0.3257]; hi_m = [0.7163, 0.6206]
axR.axhline(1.0, color=MUTED, linewidth=LW_REF, linestyle="--", zorder=2)
axR.text(0.62, 1.03, "Image contributes nothing", fontsize=FS_S, color=MUTED, va="bottom",
         ha="center")
off = 0.07
for xs_, r, lo_, hi_, col, mk, lab, ls in [
        (reg - off, r_g, lo_g, hi_g, BLIND, "o", "Grey field", "-"),
        (reg + off, r_m, lo_m, hi_m, MM, "s", "Mismatched image", "-")]:
    axR.plot(xs_, r, color=col, linewidth=LW_DATA, linestyle=ls, zorder=3)
    axR.errorbar(xs_, r, yerr=[np.subtract(r, lo_), np.subtract(hi_, r)], fmt=mk, color=col,
                 ecolor=col, elinewidth=LW_DATA, capsize=2.5, capthick=LW_DATA,
                 markersize=MS, zorder=4, label=lab)
for xi, v, side in [(0 - off, 1.0066, -1), (1 - off, 0.7099, -1), (0 + off, 0.6754, 1),
                    (1 + off, 0.4741, 1)]:
    axR.text(xi + side * 0.1, v, f"{v:.2f}", ha="left" if side > 0 else "right", va="center",
             fontsize=FS_S, color=INK, bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
axR.set_xticks(reg); axR.set_xticklabels(["All cells", "Uncapped cells"])
axR.set_xlim(-0.45, 1.45)
axR.set_ylim(0.25, 1.13); axR.set_yticks([0.25, 0.5, 0.75, 1.0])
axR.set_ylabel("Ablated / sighted rise")
axR.legend(loc="lower left", handlelength=1.6)
ptitle(axR, "b", "Both choices move the answer")
recessive(axR)
save(fig, "fig_lead.pdf")


# ============================================================ fig_concept
fig, ax = inch_canvas(4.3, 1.42)


def box(x, y, w, h, fc, ec, label, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=LW_DATA * 0.8, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.075 if sub else 0), label, ha="center",
            va="center", fontsize=FS, color=INK, zorder=4, linespacing=1.05)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.095, sub, ha="center", va="center",
                fontsize=FS_S, color=MUTED, zorder=4, style="italic")


def harrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7,
                                 color=MUTED, lw=LW_REF, zorder=2))


BH = 0.40
for yb, name, col, fc, img in [(0.86, "Sighted", SIGHTED, PALE_S, "Target\nimage"),
                               (0.12, "Blind", BLIND, "#cfcfcf", "Grey\nfield")]:
    ax.text(0.0, yb + BH + 0.07, name, fontsize=FS_T, color=col, fontweight="bold")
    box(0.0, yb, 0.62, BH, fc, col, img)
    box(0.84, yb, 1.20, BH, "#ffffff", MUTED, "Prefix tokens", "identical token ids")
    box(2.26, yb, 0.94, BH, "#ffffff", MUTED, "Continuation")
    harrow(0.63, 0.83, yb + BH / 2)
    harrow(2.05, 2.25, yb + BH / 2)
ax.text(3.30, 0.86 + BH / 2, r"$\Delta = +0.486$", fontsize=FS_T, color=SIGHTED,
        fontweight="bold", va="center")
ax.text(3.30, 0.12 + BH / 2, r"$\Delta = +0.502$", fontsize=FS_T, color=BLIND,
        fontweight="bold", va="center")
yT, yB = 0.86 + BH / 2, 0.12 + BH / 2
ax.plot([4.02, 4.07, 4.07, 4.02], [yT, yT, yB, yB], color=INK, lw=LW_REF)
ax.text(4.12, (yT + yB) / 2, r"$\Delta_{\mathrm{image}}$" "\n" r"$-0.016$", fontsize=FS,
        color=INK, va="center", ha="left", linespacing=1.3)
save(fig, "fig_concept.pdf")


# ============================================================ fig_robustness
fig, ax = plt.subplots(figsize=(TW * 0.93, 2.75))
fig.subplots_adjust(left=0.33, right=0.985, top=0.985, bottom=0.15)
rows = [
    ("Raw primary (registered)",            -0.0160, -0.0708,  0.0400, "f1"),
    ("Within-condition, two-arm",           -0.0124, -0.0673,  0.0427, "f1"),
    ("Cross-condition length match",         0.0223, -0.0190,  0.0665, "f1"),
    ("Empty continuations removed",         -0.0045, -0.0614,  0.0536, "f1"),
    ("Object-type matched",                  0.0019, -0.0488,  0.0516, "f1"),
    ("Raw primary (registered)",            -0.0348, -0.0905,  0.0192, "f2"),
    ("Within-condition, two-arm",            0.0426, -0.0041,  0.0891, "f2"),
    ("Empty continuations removed",         -0.0557, -0.1172,  0.0053, "f2"),
    ("Object-type matched",                 -0.0413, -0.0895,  0.0040, "f2"),
]
y = [9, 8, 7, 6, 5, 3, 2, 1, 0]
CI_X = 0.106
ax.axvspan(-0.10, 0.10, color=BAND, zorder=1, linewidth=0)
STYLE = {"f1": (ARCH_A, "o", ARCH_A), "f2": (ARCH_B, "s", ARCH_A)}
for (lab, p, l, h, fam), yi in zip(rows, y):
    col, mk, mec = STYLE[fam]
    ax.errorbar([p], [yi], xerr=[[p - l], [h - p]], fmt=mk, color=col, ecolor=col,
                elinewidth=LW_DATA, capsize=2.5, capthick=LW_DATA, markersize=MS - 0.5,
                markeredgecolor=mec, markeredgewidth=0.7, zorder=3)
    ax.text(CI_X, yi, f"[{sg(l, 3)}, {sg(h, 3)}]", va="center", ha="left",
            fontsize=FS_S, color=INK, zorder=4)
ax.axvline(0, color=INK, linewidth=LW_REF, linestyle="--", zorder=2)
ax.axhline(4.0, color="#9a9a9a", linewidth=LW_REF, zorder=5)
ax.set_yticks(y); ax.set_yticklabels([lab for lab, *_ in rows])
ax.tick_params(axis="y", length=0)
ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
ax.set_xlabel(r"$\Delta_{\mathrm{image}}$ (shaded: reference margin $\pm 0.10$, not registered)")
ax.set_xlim(-0.125, 0.172); ax.set_ylim(-0.65, 10.4)
ax.text(-0.121, 9.55, "LLaVA-1.5-7B", fontsize=FS_T, color=INK, fontweight="bold",
        va="baseline")
ax.text(-0.121, 3.5, "LLaVA-OV-0.5B", fontsize=FS_T, color=INK, fontweight="bold",
        va="baseline")
recessive(ax, axis="x")
ax.spines["left"].set_visible(False)
save(fig, "fig_robustness.pdf")


# ============================================================ fig_dose
fig, ax = plt.subplots(figsize=(3.05, 1.85))
steps = [0.2792, 0.1180, 0.0372, 0.0448]
xs = np.arange(4)
ax.bar(xs, steps, 0.6, color=SIGHTED, zorder=3, edgecolor=SURFACE, linewidth=LW_EDGE)
for xi, v in zip(xs, steps):
    ax.text(xi, v + 0.007, sg(v, 3), ha="center", va="bottom", fontsize=FS_S, color=INK)
ax.annotate("Step 4 > step 3\n(point estimates)", xy=(2.72, 0.026),
            xytext=(1.62, 0.165), fontsize=FS_S, color=INK, linespacing=1.15,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_REF, mutation_scale=7,
                            shrinkA=2, shrinkB=1))
ax.set_xticks(xs); ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_xlabel("Step along the realised distinct-scene ladder")
ax.set_ylabel(r"Increment in $J$")
ax.set_ylim(0, 0.32); ax.set_yticks([0, 0.1, 0.2, 0.3])
recessive(ax)
save(fig, "fig_dose.pdf")


# ============================================================ fig_sdt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(TW * 0.93, 1.95))
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.14, top=0.87, wspace=0.36)
x = np.arange(2)
d_s = [0.4861, 2.0183]; d_b = [-0.0549, 1.2918]
a1.bar(x - BW/2, d_s, BW, color=SIGHTED, zorder=3, edgecolor=SURFACE,
       linewidth=LW_EDGE, label="Sighted")
a1.bar(x + BW/2, d_b, BW, color=BLIND, zorder=3, edgecolor=SURFACE,
       linewidth=LW_EDGE, hatch="///", label="Blind")
a1.axhline(0, color=MUTED, linewidth=LW_REF, zorder=2)
for xi, v in list(zip(x - BW/2, d_s)) + list(zip(x + BW/2, d_b)):
    a1.text(xi, v + (0.06 if v >= 0 else -0.06), sg(v), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=FS_S, color=INK)
a1.set_xticks(x); a1.set_xticklabels(["One scene", "Five scenes"])
a1.set_xlim(-0.55, 1.55)
a1.set_ylabel("Sensitivity $d$\u2032"); a1.set_ylim(-0.55, 2.45)
ptitle(a1, "a", "$d$\u2032 rises with scene count, blind or not")
a1.legend(loc="upper left")
recessive(a1)

c_s = [-0.2130, 0.2371]; c_b = [0.0532, 0.0975]
a2.bar(x - BW/2, c_s, BW, color=SIGHTED, zorder=3, edgecolor=SURFACE, linewidth=LW_EDGE)
a2.bar(x + BW/2, c_b, BW, color=BLIND, zorder=3, edgecolor=SURFACE,
       linewidth=LW_EDGE, hatch="///")
a2.axhline(0, color=MUTED, linewidth=LW_REF, zorder=2)
for xi, v in list(zip(x - BW/2, c_s)) + list(zip(x + BW/2, c_b)):
    a2.text(xi, v + (0.015 if v >= 0 else -0.015), sg(v), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=FS_S, color=INK)
a2.set_xticks(x); a2.set_xticklabels(["One scene", "Five scenes"])
a2.set_xlim(-0.55, 1.55)
a2.set_ylabel(r"Criterion $c$"); a2.set_ylim(-0.33, 0.36)
ptitle(a2, "b", r"$c$ moves $+0.45$ sighted, $+0.04$ blind")
recessive(a2)
save(fig, "fig_sdt.pdf")


# ============================================================ fig_forced
fig, ax = plt.subplots(figsize=(2.5, 1.95))
vals = [0.4859, -0.0095]
los  = [0.4283, -0.0314]
his  = [0.5417,  0.0124]
xs = np.arange(2)
cols = [READ_A, READ_B]
err = [[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]]
ax.bar(xs, vals, 0.52, color=cols, zorder=3, edgecolor=READ_A, linewidth=LW_EDGE)
ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=LW_DATA,
            capsize=2.5, capthick=LW_DATA, zorder=4)
ax.axhline(0, color=MUTED, linewidth=LW_REF, zorder=2)
for xi, v, h in zip(xs, vals, his):
    ax.text(xi, h + 0.025, sg(v), ha="center", va="bottom", fontsize=FS_S, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels(["Free generation", "Forced choice"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel(r"Scene-count contrast in $J$")
ax.set_ylim(-0.1, 0.64); ax.set_yticks([0, 0.2, 0.4, 0.6])
recessive(ax)
save(fig, "fig_forced.pdf")


# ============================================================ fig_zroc
# Operating points in z-coordinates.  d' = z(H) - z(F) is height above the chance diagonal, so
# lines of constant d' are parallel to it and a pure threshold move slides along them.
# Rates: appendix tab:percond (prefix endpoint, LLaVA-1.5-7B) and tab:mdlevels (CHAIR, PAI).
from scipy.stats import norm as _norm
zz = _norm.ppf
fig, (a1, a2) = plt.subplots(1, 2, figsize=(TW * 0.93, 2.3),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.9, wspace=0.3)


def isod(ax, xs, ds):
    for d in ds:
        ax.plot(xs, xs + d, color=MUTED, linewidth=LW_GRID + 0.15, linestyle=":", zorder=1)


def dlab(ax, x, d, text=None, off=0.02, below=False):
    ax.text(x, x + d + (-off if below else off), text or f"$d$\u2032 = {d:g}", fontsize=FS_S,
            color=MUTED, rotation=45, rotation_mode="anchor", ha="left",
            va="top" if below else "bottom", clip_on=True)


def zarrow(ax, p0, p1, col, ls, lab, lab_xy):
    ax.annotate("", xy=p1, xytext=p0, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=LW_DATA + 0.3, linestyle=ls,
                                shrinkA=2.5, shrinkB=2.5, mutation_scale=9))
    ax.plot(*p0, "o", mfc="white", mec=col, mew=LW_DATA, ms=MS, zorder=5)
    ax.plot(*p1, "o", color=col, ms=MS, zorder=5)
    ax.text(*lab_xy, lab, fontsize=FS, color=col, ha="left", va="center")


S0 = (zz(0.4880), zz(0.6761)); S1 = (zz(0.1063), zz(0.7803))
B0 = (zz(0.4897), zz(0.4678)); B1 = (zz(0.2285), zz(0.7085))
isod(a1, np.linspace(-2.0, 0.5, 50), [0, 1, 2])
dlab(a1, 0.12, 0, "Chance", off=0.08)
dlab(a1, -1.30, 1, off=0.08); dlab(a1, -1.84, 2, off=0.08)
zarrow(a1, S0, S1, SIGHTED, "-",  "Sighted", (-0.80, 0.78))
zarrow(a1, B0, B1, BLIND,   "--", "Blind",   (-1.07, 0.38))
a1.set_xlim(-1.9, 0.35); a1.set_ylim(-0.45, 1.2)
a1.set_aspect("equal", adjustable="box")
a1.set_xlabel("$z(F)$"); a1.set_ylabel("$z(H)$")
ptitle(a1, "a", r"Prefix endpoint, one $\rightarrow$ five scenes")
recessive(a1)

V = (zz(0.0100), zz(0.7816)); P = (zz(0.0050), zz(0.7083))
isod(a2, np.linspace(-2.8, -2.0, 20), [2.9, 3.1, 3.3])
dlab(a2, -2.29, 2.9, off=0.03); dlab(a2, -2.235, 3.1, off=0.03, below=True)
dlab(a2, -2.66, 3.3, off=0.03)
zarrow(a2, V, P, INK, "-", r"Vanilla $\rightarrow$ PAI", (-2.37, 0.45))
a2.set_xlim(-2.75, -2.06); a2.set_ylim(0.40, 0.98)
a2.set_xticks([-2.6, -2.4, -2.2])
a2.set_aspect("equal", adjustable="box")
a2.set_xlabel("$z(F)$"); a2.set_ylabel("$z(H)$")
ptitle(a2, "b", "PAI on CHAIR")
recessive(a2)
# equal-aspect panels of different heights: pin both titles to one baseline
fig.canvas.draw()
ytop = max(a1.get_position().y1, a2.get_position().y1)
for ax_ in (a1, a2):
    t = ax_.title
    bb = ax_.get_position()
    t.set_transform(fig.transFigure); t.set_position((bb.x0, ytop + 0.03)); t.set_ha("left")
save(fig, "fig_zroc.pdf")


# ============================================================ fig_hf
# The H/F decomposition of PAI's CHAIR gain (appendix tab:mdlevels / tab:mddecomp; PAI
# alpha=0.5 vs vanilla, LLaVA-1.5-7B, 500 images).  Boxes are fitted to their text, so no
# line can cross a border.
fig, ax = inch_canvas(4.9, 1.72)
fig.canvas.draw()


def text_panel(xc, ytop, title, body, edge, fc="white"):
    t1 = ax.text(xc, ytop, title, ha="center", va="top", fontsize=FS_T, color=INK,
                 fontweight="bold", zorder=3)
    t2 = ax.text(xc, ytop - 0.2, body, ha="center", va="top", fontsize=FS, color=INK,
                 zorder=3, linespacing=1.25)
    return fit_box(ax, [t1, t2], 0.12, 0.07, facecolor=fc, edgecolor=edge,
                   linewidth=LW_DATA)


def varrow(x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=LW_DATA,
                                mutation_scale=9, shrinkA=0, shrinkB=0))


top = text_panel(2.45, 1.64, r"$\mathbf{CHAIR}_{\mathbf{i}}$ improves: $12.8 \rightarrow 7.2$",
                 "A published method, scored the standard way", INK, fc=BAND)
left = text_panel(1.2, 1.02, r"$H$ falls: $0.78 \rightarrow 0.71$",
                  "It names fewer objects that\n" r"$are$ in the image" "\n(seven points of recall)",
                  SIGHTED)
right = text_panel(3.7, 1.02, r"$F$ falls: $0.010 \rightarrow 0.005$",
                   "It names fewer objects that\n" r"are $not$ in the image" "\n(the half CHAIR rewards)",
                   BLIND)
yb = top[1]
ym = (yb + left[3]) / 2
ax.plot([1.2, 3.7], [ym, ym], color=MUTED, lw=LW_DATA, zorder=1)
ax.plot([2.45, 2.45], [yb, ym], color=MUTED, lw=LW_DATA, zorder=1)
varrow(1.2, ym, left[3]); varrow(3.7, ym, right[3])
ax.text(2.45, left[1] - 0.06, r"Threshold $c$: $+0.24$ [$+0.20$, $+0.28$]   $\cdot$   "
        "Separation: registered primary cannot resolve",
        ha="center", va="top", fontsize=FS_S, color=MUTED)
save(fig, "fig_hf.pdf")

print("wrote:", ", ".join(sorted(os.listdir(OUT))))
