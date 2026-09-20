#!/usr/bin/env python3
"""
Figures for "When Grounding Metrics Survive Blinding".

Every number is transcribed from the filed result artefacts (wia_dprime.json,
wia_baseline.json, wia_mrate.json, wia_xklen_cross_20554.json, wia_bench2).
This script only draws; it computes nothing that is reported as a result.

Palette validated colourblind-safe before use (protan dE 23.3, normal dE 30.3).
Blue/orange means sighted/blind and nothing else; figures whose two series are
not sighted/blind carry their own pair (see the colour-semantics block below).
Every pairing also has a non-colour encoding -- hatch, marker shape, row label,
block rule -- so the figures survive greyscale print.
Figures are drawn at the width they are placed at in the document, so no figure
is scaled down and no label lands below ~7pt on the page.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# --- colour semantics -------------------------------------------------------
# SIGHTED/BLIND (blue/orange) is reserved for the sighted-vs-blind contrast and
# is used ONLY in fig_concept, fig_paradox(a) and fig_sdt.  Figures whose two
# series are not sighted/blind get their own pair, so the blue/orange pair never
# stands for anything else:
#   fig_robustness  architecture 1 vs architecture 2  -> ARCH_A / ARCH_B (plum)
#   fig_forced      free generation vs forced choice  -> READ_A / READ_B (green)
# Both replacement pairs are lightness-ramped within one hue, so the separation
# survives dichromacy and greyscale.  Measured (CIEDE2000, Vienot dichromat sim)
# against the validated blue/orange baseline (normal 49.1, protan 57.3, dL* 12.2):
#   ARCH plum  (#6B2D5B/#C98FB4) normal 36.7 protan 36.0 deutan 36.8 tritan 36.0 dL* 37.2
#   READ green (#2A6B3C/#98C9A3) normal 33.2 protan 31.9 deutan 33.9 tritan 33.0 dL* 36.6
# i.e. every axis clears the validated floor (normal 30.3 / protan 23.3) and the
# greyscale separation is ~3x the baseline's.
SIGHTED = "#1F5FA9"
BLIND   = "#C4620A"
INK     = "#1a1a1a"
MUTED   = "#6b6b6b"
GRID    = "#d8d8d4"
SURFACE = "#ffffff"          # page white: figures must not print an off-white panel
PALE_S  = "#dbe6f3"          # sighted-tinted fill, fig_concept only
BAND    = "#e9e9e6"          # neutral equivalence-margin shading (not blue)
CITEXT  = "#4a4a4a"
ARCH_A  = "#6B2D5B"
ARCH_B  = "#C98FB4"
READ_A  = "#2A6B3C"
READ_B  = "#98C9A3"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.edgecolor": "none",
})
W = 5.5
BW = 0.34


def recessive(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ============================================================ fig_concept
# Canvas cropped to the diagram now that the in-figure footnote is gone; the
# height is cut in the same proportion as the y-range so the boxes and type keep
# exactly the scale they had.
fig, ax = plt.subplots(figsize=(W, 1.77))
ax.set_xlim(0, 11); ax.set_ylim(1.02, 4.78); ax.axis("off")


def box(x, y, w, h, fc, ec, label, sub=None, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=0.9, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.16 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.26, sub, ha="center", va="center",
                fontsize=6.5, color=MUTED, zorder=4, style="italic")


ax.text(0.05, 4.35, "SIGHTED", fontsize=7.5, color=SIGHTED, fontweight="bold")
box(0.05, 3.25, 1.5, 0.95, PALE_S, SIGHTED, "target\nimage", fs=7)
box(1.85, 3.25, 3.1, 0.95, "#ffffff", MUTED, "prefix tokens", "identical ids", fs=7.5)
box(5.25, 3.25, 1.9, 0.95, "#ffffff", MUTED, "continuation", fs=7.5)
ax.text(7.45, 3.72, r"$\Delta = +0.486$", fontsize=8.5, color=SIGHTED,
        fontweight="bold", va="center")

ax.text(0.05, 2.30, "BLIND", fontsize=7.5, color=BLIND, fontweight="bold")
box(0.05, 1.20, 1.5, 0.95, "#c9c9c9", BLIND, "grey\nrectangle", fs=7)
box(1.85, 1.20, 3.1, 0.95, "#ffffff", MUTED, "prefix tokens", "identical ids", fs=7.5)
box(5.25, 1.20, 1.9, 0.95, "#ffffff", MUTED, "continuation", fs=7.5)
ax.text(7.45, 1.67, r"$\Delta = +0.502$", fontsize=8.5, color=BLIND,
        fontweight="bold", va="center")

for y in (3.72, 1.67):
    ax.add_patch(FancyArrowPatch((1.62, y), (1.80, y), arrowstyle="->",
                                 mutation_scale=8, color=MUTED, lw=0.8))
    ax.add_patch(FancyArrowPatch((5.02, y), (5.20, y), arrowstyle="->",
                                 mutation_scale=8, color=MUTED, lw=0.8))

ax.plot([9.85, 9.85], [1.67, 3.72], color=INK, lw=0.9)
ax.plot([9.72, 9.85], [3.72, 3.72], color=INK, lw=0.9)
ax.plot([9.72, 9.85], [1.67, 1.67], color=INK, lw=0.9)
ax.text(10.0, 2.70, r"$\Delta_{\mathrm{image}}$" "\n" r"$-0.016$", fontsize=7.5,
        color=INK, va="center")

# The in-figure footnote that used to sit here duplicated the LaTeX caption
# almost verbatim; it is deleted and the canvas cropped to the diagram, which
# also returns the vertical space it occupied.
save(fig, "fig_concept.pdf")


# ============================================================ fig_lead
# The paper's lead figure (Figure 1).  Two decimals throughout, matching the main text.
# Panel (a) doubles as the positive control: the image raises J by about +0.2 WITHIN each
# condition (intervals from wia_dprime.json, lanes/F1/ci/sight_gap_cap{1,5}), and raises it by
# the same amount in both, which is why it contributes nothing to the one->five contrast.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(W, 1.95),
                               gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.62})
x = np.arange(2)
g_s = [0.18814, 0.67401]
g_b = [-0.02191, 0.47997]
axL.bar(x - BW/2, g_s, BW, color=SIGHTED, label="sighted   (rise $+0.49$)",
        zorder=3, edgecolor=SURFACE, linewidth=1.0)
axL.bar(x + BW/2, g_b, BW, color=BLIND, label="blind   (rise $+0.50$)",
        zorder=3, edgecolor=SURFACE, linewidth=1.0, hatch="///")
axL.axhline(0, color=MUTED, linewidth=0.7, zorder=2)
for xi, v in list(zip(x - BW/2, g_s)) + list(zip(x + BW/2, g_b)):
    axL.text(xi, v + (0.025 if v >= 0 else -0.075), f"{v:+.2f}", ha="center",
             fontsize=7, color=INK)
# within-condition image effect, sighted minus blind
for xc, top, lab in [(0, 0.188, "image: $+0.21$"), (1, 0.674, "image: $+0.19$")]:
    axL.text(xc, top + 0.13, lab, ha="center", fontsize=6.8, color=MUTED, style="italic")
axL.set_xticks(x); axL.set_xticklabels(["one scene", "five scenes"])
axL.set_ylabel(r"$J = H - F$")
axL.set_ylim(-0.14, 1.26)
axL.set_title("(a) the score, with and without the image", loc="left")
axL.legend(frameon=False, loc="upper left", handlelength=1.2, ncol=1,
           borderaxespad=0.2, labelspacing=0.3, fontsize=7)
recessive(axL)

# Four-architecture run (all measured together); Qwen2.5-VL is reported in the appendix only,
# because its four-token median continuation leaves the endpoint almost nothing to measure.
labels = ["LLaVA-1.5-7B", "LLaVA-OV-0.5B", "Kosmos-2", "Qwen3-VL-8B"]
pt = [-0.0160, -0.0348, -0.1028, +0.0598]
lo = [-0.0701, -0.0906, -0.1677, +0.0235]
hi = [+0.0399, +0.0179, -0.0360, +0.0963]
y = np.arange(4)[::-1]
err = [[p - l for p, l in zip(pt, lo)], [h - p for p, h in zip(pt, hi)]]
axR.axvspan(-0.10, 0.10, color=BAND, alpha=0.9, zorder=1)
for xi, yi, e0, e1 in zip(pt, y, err[0], err[1]):
    axR.errorbar([xi], [yi], xerr=[[e0], [e1]], fmt="o", color=INK, ecolor=INK,
                 elinewidth=1.0, capsize=3, markersize=4.5, zorder=3)
axR.axvline(0, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
for xi, yi in zip(pt, y):
    axR.text(xi, yi + 0.22, f"{xi:+.2f}", ha="center", fontsize=7.0, color=INK)
axR.set_yticks(list(y) + [-0.95])
axR.set_yticklabels(labels + ["mismatched image\n(LLaVA-1.5-7B)"], fontsize=7.4)
for lab in axR.get_yticklabels()[-1:]:
    lab.set_color(SIGHTED); lab.set_fontsize(6.6)
axR.set_ylim(-0.6, 3.6); axR.set_xlim(-0.20, 0.15)
axR.set_xlabel(r"$\Delta_{\mathrm{image}}$   (shaded: $\pm 0.10$)")
axR.errorbar([0.1598], [-0.95], xerr=[[0.1598 - 0.1052], [0.2129 - 0.1598]], fmt="s",
             color=SIGHTED, ecolor=SIGHTED, elinewidth=1.0, capsize=3, markersize=4.5, zorder=3)
axR.text(0.1598, -0.73, "$+0.16$", ha="center", fontsize=7.0, color=SIGHTED)

axR.set_ylim(-1.7, 3.6); axR.set_xlim(-0.20, 0.25)
axR.set_title("(b) the part of the rise due to the image", loc="left")
recessive(axR, axis="x")
save(fig, "fig_lead.pdf")


# ============================================================ fig_dose
fig, ax = plt.subplots(figsize=(W * 0.58, 1.72))
steps = [0.2792, 0.1180, 0.0372, 0.0448]
xs = np.arange(4)
ax.bar(xs, steps, 0.6, color=SIGHTED, zorder=3, edgecolor=SURFACE, linewidth=1.0)
for xi, v in zip(xs, steps):
    ax.text(xi, v + 0.008, f"{v:+.4f}", ha="center", fontsize=7, color=INK)
# The arrow lands on the left flank of bar 4, below its "+0.0448" data label,
# so the arrowhead no longer strikes through the number it points at.
ax.annotate("step 4 $>$ step 3\n(point estimates)", xy=(2.715, 0.0255),
            xytext=(1.78, 0.150), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                            shrinkA=2, shrinkB=1))
ax.set_xticks(xs); ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_xlabel("step along the realised distinct-scene ladder")
ax.set_ylabel(r"increment in $J$")
ax.set_ylim(0, 0.32)
ax.set_title(r"monotone and front-loaded ($\rho_S = +1.0$)", loc="left")
recessive(ax)
save(fig, "fig_dose.pdf")


# ============================================================ fig_sdt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.05))
x = np.arange(2)

d_s = [0.4861, 2.0183]; d_b = [-0.0549, 1.2918]
a1.bar(x - BW/2, d_s, BW, color=SIGHTED, zorder=3, edgecolor=SURFACE,
       linewidth=1.0, label="sighted")
a1.bar(x + BW/2, d_b, BW, color=BLIND, zorder=3, edgecolor=SURFACE,
       linewidth=1.0, hatch="///", label="blind")
a1.axhline(0, color=MUTED, linewidth=0.7, zorder=2)
for xi, v in list(zip(x - BW/2, d_s)) + list(zip(x + BW/2, d_b)):
    a1.text(xi, v + (0.07 if v >= 0 else -0.19), f"{v:+.2f}", ha="center",
            fontsize=6.8, color=INK)
a1.set_xticks(x); a1.set_xticklabels(["one scene", "five scenes"])
a1.set_ylabel(r"sensitivity $d'$"); a1.set_ylim(-0.6, 2.45)
a1.set_title(r"(a) $d'$ rises with scene count, blind or not", loc="left")
a1.legend(frameon=False, loc="upper left", handlelength=1.3)
recessive(a1)

c_s = [-0.2130, 0.2371]; c_b = [0.0532, 0.0975]
a2.bar(x - BW/2, c_s, BW, color=SIGHTED, zorder=3, edgecolor=SURFACE, linewidth=1.0)
a2.bar(x + BW/2, c_b, BW, color=BLIND, zorder=3, edgecolor=SURFACE,
       linewidth=1.0, hatch="///")
a2.axhline(0, color=MUTED, linewidth=0.7, zorder=2)
for xi, v in list(zip(x - BW/2, c_s)) + list(zip(x + BW/2, c_b)):
    a2.text(xi, v + (0.022 if v >= 0 else -0.055), f"{v:+.2f}", ha="center",
            fontsize=6.8, color=INK)
a2.set_xticks(x); a2.set_xticklabels(["one scene", "five scenes"])
a2.set_ylabel(r"criterion $c$"); a2.set_ylim(-0.32, 0.36)
a2.set_title(r"(b) $c$ moves $+0.45$ sighted, $+0.04$ blind", loc="left")
recessive(a2)
save(fig, "fig_sdt.pdf")


# ============================================================ fig_forced
fig, ax = plt.subplots(figsize=(W * 0.62, 1.95))
vals = [0.4859, -0.0095]
los  = [0.4283, -0.0314]
his  = [0.5417,  0.0124]
xs = np.arange(2)
# free generation vs forced choice is a readout contrast, not sighted/blind, so
# it gets its own lightness-ramped green pair; the x tick labels name the arms.
cols = [READ_A, READ_B]
err = [[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]]
ax.bar(xs, vals, 0.5, color=cols, zorder=3, edgecolor=READ_A, linewidth=0.8)
ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1.0,
            capsize=3, zorder=4)
ax.axhline(0, color=MUTED, linewidth=0.7, zorder=2)
for xi, v, h in zip(xs, vals, his):
    ax.text(xi, h + 0.03, f"{v:+.4f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels(["free generation\n(model selects objects)",
                    "forced choice\n(question supplies object)"])
ax.set_ylabel(r"scene-count contrast in $J$")
ax.set_ylim(-0.12, 0.64)
ax.set_title("same stimuli, same model, same units", loc="left")
recessive(ax)
save(fig, "fig_forced.pdf")


# ============================================================ fig_robustness
# Canvas sized to the width it is placed at (0.92 x 5.5in text block) so the
# figure is not scaled down in the document: at the old 5.5in canvas the tight
# bbox came out 5.77in and everything shrank by 12%, which is what dropped the
# CI strings to ~5.8pt on the page.
fig, ax = plt.subplots(figsize=(5.02, 2.95))
fig.subplots_adjust(left=0.300, right=0.988, top=0.972, bottom=0.160)
rows = [
    ("raw primary (registered)",            -0.0160, -0.0708,  0.0400, "f1"),
    ("within-condition, two-arm",           -0.0124, -0.0673,  0.0427, "f1"),
    ("cross-condition length match",         0.0223, -0.0190,  0.0665, "f1"),
    ("empty continuations removed",         -0.0045, -0.0614,  0.0536, "f1"),
    ("object-type matched",                  0.0019, -0.0488,  0.0516, "f1"),
    ("raw primary (registered)",            -0.0348, -0.0905,  0.0192, "f2"),
    ("within-condition, two-arm",            0.0426, -0.0041,  0.0891, "f2"),
    ("empty continuations removed",         -0.0557, -0.1172,  0.0053, "f2"),
    ("object-type matched",                 -0.0413, -0.0895,  0.0040, "f2"),
]
# Every row of the appendix truncation table for the two LLaVA architectures is plotted.
# A two-row gap between the blocks carries the separating rule and the second
# block heading without either touching a data row.
y = [9, 8, 7, 6, 5, 3, 2, 1, 0]
CI_X = 0.104                      # CI column starts clear of the shaded band
ax.axvspan(-0.10, 0.10, color=BAND, alpha=0.9, zorder=1)
# Two architectures reuse the same row labels, so the grouping must survive
# greyscale: a rule between the blocks, a bold block heading over each, and a
# different marker shape per block.  Colour is the fourth, redundant cue.
STYLE = {"f1": (ARCH_A, "o", ARCH_A), "f2": (ARCH_B, "s", ARCH_A)}
for (lab, p, l, h, fam), yi in zip(rows, y):
    col, mk, mec = STYLE[fam]
    ax.errorbar([p], [yi], xerr=[[p - l], [h - p]], fmt=mk, color=col,
                ecolor=col, elinewidth=1.1, capsize=3, markersize=4.5,
                markeredgecolor=mec, markeredgewidth=0.7, zorder=3)
    ax.text(CI_X, yi, f"[{l:+.4f}, {h:+.4f}]", va="center", ha="left",
            fontsize=7.2, color=CITEXT, zorder=4)
ax.axvline(0, color=INK, linewidth=0.9, linestyle="--", zorder=2)
ax.axhline(4.05, color="#8e8e8e", linewidth=0.8, zorder=5)
ax.set_yticks(y); ax.set_yticklabels([lab for lab, *_ in rows])
ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])   # no gridline through the CI column
ax.set_xlabel(r"$\Delta_{\mathrm{image}}$  (shaded: reference margin $\pm 0.10$, not registered)")
ax.set_xlim(-0.125, 0.176); ax.set_ylim(-0.70, 10.15)
ax.text(-0.122, 9.48, "A.  LLaVA-1.5-7B", fontsize=7.5, color=INK,
        fontweight="bold", va="baseline")
ax.text(-0.122, 3.42, "B.  LLaVA-OV-0.5B", fontsize=7.5, color=INK,
        fontweight="bold", va="baseline")
recessive(ax, axis="x")
save(fig, "fig_robustness.pdf")

print("wrote:", ", ".join(sorted(os.listdir(OUT))))

# ============================================================ fig_zroc
# Operating points in z-coordinates.  d' = z(H) - z(F) is height above the chance diagonal, so
# lines of constant d' are parallel to it and a pure threshold move slides along them.
# Rates: appendix tab:percond (prefix endpoint, LLaVA-1.5-7B) and tab:mdlevels (CHAIR, PAI).
from scipy.stats import norm as _norm
zz = _norm.ppf
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.35),
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.42})

def isod(ax, xs, ds):
    for d in ds:
        ax.plot(xs, xs + d, color=MUTED, linewidth=0.6, linestyle=":", zorder=1)

def dlab(ax, x, d, text=None, off=0.02):
    ax.text(x, x + d + off, text or f"$d'={d:g}$", fontsize=6.6, color=MUTED, rotation=45,
            rotation_mode="anchor", ha="left", va="bottom", clip_on=True)

def arrow(ax, p0, p1, col, ls, lab, lab_xy):
    ax.annotate("", xy=p1, xytext=p0, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, linestyle=ls,
                                shrinkA=2, shrinkB=2, mutation_scale=9))
    ax.plot(*p0, "o", mfc="white", mec=col, ms=4.2, zorder=5)
    ax.plot(*p1, "o", color=col, ms=4.2, zorder=5)
    ax.text(*lab_xy, lab, fontsize=7.2, color=col, ha="left", va="center", clip_on=True)

# (a) prefix endpoint, one scene -> five scenes (open marker = one scene, filled = five)
S0 = (zz(0.4880), zz(0.6761)); S1 = (zz(0.1063), zz(0.7803))
B0 = (zz(0.4897), zz(0.4678)); B1 = (zz(0.2285), zz(0.7085))
isod(a1, np.linspace(-2.0, 0.5, 50), [0, 1, 2])
dlab(a1, 0.00, 0, "chance", off=0.10)
dlab(a1, -1.32, 1, off=0.10); dlab(a1, -1.82, 2, off=0.10)
arrow(a1, S0, S1, SIGHTED, "-",  "sighted", (-1.22, 0.93))
arrow(a1, B0, B1, BLIND,   "--", "blind",   (-1.10, 0.38))
a1.set_xlim(-1.9, 0.35); a1.set_ylim(-0.45, 1.2)
a1.set_aspect("equal", adjustable="box")
a1.set_xlabel("$z(F)$"); a1.set_ylabel("$z(H)$")
a1.set_title("(a) prefix endpoint, one $\\rightarrow$ five scenes", loc="left")
recessive(a1)

# (b) standard CHAIR, vanilla -> PAI
V = (zz(0.0100), zz(0.7816)); P = (zz(0.0050), zz(0.7083))
isod(a2, np.linspace(-2.8, -2.0, 20), [2.9, 3.1, 3.3])
dlab(a2, -2.31, 2.9, off=0.035); dlab(a2, -2.28, 3.1, off=0.035); dlab(a2, -2.73, 3.3, off=0.035)
arrow(a2, V, P, INK, "-", "vanilla $\\rightarrow$ PAI", (-2.74, 0.47))
a2.set_xlim(-2.75, -2.06); a2.set_ylim(0.40, 0.98)
a2.set_aspect("equal", adjustable="box")
a2.set_xlabel("$z(F)$"); a2.set_ylabel("$z(H)$")
a2.set_title("(b) PAI on CHAIR", loc="left")
recessive(a2)
save(fig, "fig_zroc.pdf")

# ============================================================ fig_hf
# Prof. Park's Priority 3: make the H/F decomposition the star.  Two rows only; the "one number,
# two behaviours" reading lives in the caption so the float stays inside the page budget.
# Numbers: appendix tab:mdlevels / tab:mddecomp (PAI alpha=0.5 vs vanilla, LLaVA-1.5-7B, 500 images).
fig, ax = plt.subplots(figsize=(W, 1.62))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.05); ax.axis("off")

def panel(x, y, w, h, title, body, edge=INK, fc="white"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=edge, linewidth=1.0, zorder=2))
    ax.text(x + w/2, y + h - 0.33, title, ha="center", va="top", fontsize=8.0, color=INK,
            fontweight="bold", zorder=3)
    ax.text(x + w/2, y + h - 0.83, body, ha="center", va="top", fontsize=7.2, color=INK,
            zorder=3, linespacing=1.35)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, mutation_scale=11))

panel(2.55, 3.72, 4.9, 1.28, r"$\mathrm{CHAIR}_i$ improves:  $12.8 \rightarrow 7.2$",
      "a published method, scored the standard way", fc=BAND)
ax.plot([2.45, 7.55], [3.34, 3.34], color=MUTED, lw=1.0, zorder=1)
arrow(5.0, 3.72, 5.0, 3.36)
arrow(2.45, 3.34, 2.45, 2.86)
arrow(7.55, 3.34, 7.55, 2.86)

panel(0.30, 0.92, 4.30, 1.92, r"$H$ falls:  $0.78 \rightarrow 0.71$",
      "it names fewer objects that\nARE in the image\n(seven points of recall)", edge=SIGHTED)
panel(5.40, 0.92, 4.30, 1.92, r"$F$ falls:  $0.010 \rightarrow 0.005$",
      "it names fewer objects that\nare NOT in the image\n(the half CHAIR rewards)", edge=BLIND)

ax.text(5.0, 0.16, r"separation $d'$ $+0.02$ [$-0.06$, $+0.09$], unchanged   "
                   r"$\cdot$   threshold $c$ $+0.24$ [$+0.20$, $+0.28$]",
        ha="center", va="bottom", fontsize=7.2, color=MUTED)
save(fig, "fig_hf.pdf")
