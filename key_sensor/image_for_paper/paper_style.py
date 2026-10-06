"""Shared look-and-feel for the ICRA figures in this folder.

Every figure here goes through `paper_rc()` and `style_axes()`, so editing a
font size, a column width or a colour in this file changes it everywhere.

Geometry: a figure saved at `PAGE` inches wide drops into \\begin{figure*} at
scale 1.0, so the point sizes below are the point sizes that end up on paper.
IEEEtran sets body text at 10pt and captions at 8pt; figure text sits between.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Column geometry (inches) ─────────────────────────────────────────────────
COL = 3.45    # \begin{figure}   single column
PAGE = 7.16   # \begin{figure*}  full page width

# Standard heights, so figures stack consistently in the paper.
H_SHORT = 2.6   # one row of bars
H_TALL = 3.0    # two stacked panels (e.g. an UpSet matrix under its bars)

RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,    # in-axes annotations
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    9,
    "axes.linewidth":     0.7,
    "pdf.fonttype":       42,   # embed real text (Type 42), not outlines
    "ps.fonttype":        42,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}


def paper_rc(**overrides):
    """`with paper_rc():` — applies RC without mutating global rcParams."""
    return plt.rc_context({**RC, **overrides})


def style_axes(ax, *, grid="y", spines=("left", "bottom")):
    """Dashed grid on one axis, hidden spines elsewhere, thin ticks."""
    for axis, on in ((ax.xaxis, grid in ("x", "both")), (ax.yaxis, grid in ("y", "both"))):
        # line properties must not be passed when disabling, or mpl re-enables it
        if on:
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.40, zorder=0)
        else:
            axis.grid(False)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.7)


def grouped_bars(ax, groups, series, values, errors=None, colors=None,
                 *, width=0.80, bar_gap=0.92):
    """One cluster of bars per group, one bar per series within the cluster.

    `values` and `errors` are keyed by (group, series). Returns the x centre of
    every bar as {(group, series): x}, so callers can annotate above them.
    """
    bar_w = width / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * bar_w
    colors = colors if colors is not None else [None] * len(series)
    centres = {}

    for s, offset, color in zip(series, offsets, colors):
        xs = np.arange(len(groups)) + offset
        ys = [values[(g, s)] for g in groups]
        ax.bar(xs, ys, width=bar_w * bar_gap, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5, zorder=3)
        if errors is not None:
            ax.errorbar(xs, ys, yerr=[errors[(g, s)] for g in groups], fmt="none",
                        ecolor="#222222", elinewidth=0.9, capsize=1.6,
                        capthick=0.9, zorder=5)
        centres.update({(g, s): x for g, x in zip(groups, xs)})

    return centres


def panel_label(ax, text, *, x=-0.22, y=1.02):
    """Bold "(a)" / "(b)" above the top-left corner of a panel."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")


def swatch_legend(fig, labels, colors, *, y=-0.22, alpha=0.85):
    """Horizontal colour-swatch legend centred under the axes."""
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=alpha, edgecolor="none")
               for c in colors]
    return fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                      frameon=False, bbox_to_anchor=(0.5, y),
                      handlelength=1.2, handletextpad=0.4, columnspacing=1.0)


def save_figure(fig, stem, *, pdf=True, png=True, jpg=True):
    """Write <stem>.pdf / .png beside the notebook using the RC savefig settings."""
    written = []
    for want, ext in ((pdf, "pdf"), (png, "png"), (jpg, "jpg")):
        if want:
            fig.savefig(f"{stem}.{ext}")
            written.append(f"{stem}.{ext}")
    print("Saved -> " + ", ".join(written))


# ── Sensor sets compared in the local-proprioception figures ────────────────
SENSORS = ["pos_action", "vel", "vel_action", "pos_vel_action"]
SENSOR_LABEL = {
    "pos_action":     "pos+act",
    "vel":            "vel",
    "vel_action":     "vel+act",
    "pos_vel_action": "pos+vel+act",
}
# seaborn "muted", spelled out so the palette is visible and editable here.
SENSOR_COLOR = {
    "pos_action":     "#4878d0",
    "vel":            "#ee854a",
    "vel_action":     "#6acc64",
    "pos_vel_action": "#d65f5f",
}

TERRAIN_LABEL = {
    "flat":     "Flat",
    "rough":    "Rough",
    "morph":    "Morph",
    "slope10":  "Slope Downhill",
    "slope_10": "Slope Uphill",
    "mass1000": "Added mass",
}

# ── Modalities ablated in the sensor-loss figure (Okabe-Ito, colour-blind safe) ──
ABLATIONS = ["pos", "vel", "action", "fc", "IMU"]
ABLATION_LABEL = {
    "pos":      "No Position",
    "vel":      "No Velocity",
    "action":   "No Action",
    "fc":       "No Foot\nContact",
    "IMU":      "No IMU",
    "baseline": "No\nAblation",
}
ABLATION_COLOR = {
    "pos":      "#0072B2",
    "vel":      "#E69F00",
    "action":   "#009E73",
    "fc":       "#CC79A7",
    "IMU":      "#D55E00",
    "baseline": "#555555",
}
