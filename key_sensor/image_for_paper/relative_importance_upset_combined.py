"""Relative sensory-feedback importance and training completeness.

Two panels of one `figure*` (IEEE double-column width):

- **(a)** integrated-gradients relative importance per sensor modality
  (Yu et al., Nat. Mach. Intell. 2023 -- arXiv:2306.17101), bar = mean over
  the 10 "flat" DAgger students, dots = individual policies. Computed by
  ../../integrate_grad/relative_importance.ipynb from the real on-policy IG
  results in ../../integrate_grad/ig/STUDENT-dagger-{0..9}-ig/ig_summary.json.
- **(b)** an UpSet plot of how many policies ever learned to walk from each
  sensor subset during training.

Both bar axes are placed with the same GridSpec top/bottom, so their x axes
(baselines) sit at the same level and the two panels stand the same height --
mirroring sensor_loss_upset_plot.ipynb, whose panel (b) this reuses unchanged.

Figure geometry and fonts come from `paper_style.py`.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from paper_style import (ABLATION_COLOR, PAGE, H_TALL,
                         panel_label, paper_rc, save_figure, style_axes)

IG_ROOT = Path(__file__).resolve().parents[2] / "integrate_grad"

# ── Panel (a) data: relative importance per sensor modality ─────────────────
R = np.load(IG_ROOT / "relative_importance.npy")  # (num_policies, 5) fractions, sum=1 per row
meta = json.loads((IG_ROOT / "relative_importance_meta.json").read_text())
group_order = meta["group_order"]  # ["pos","vel","action","imu","fc"]

RI_ORDER = ["pos", "vel", "action", "fc", "imu"]
# RI_LABEL = {"pos": "Position", "vel": "Velocity", "action": "Action",
#             "fc": "Foot\nContact", "imu": "IMU"}
RI_LABEL = {"pos": "Pos", "vel": "Vel", "action": "Act",
               "imu": "IMU", "fc": "FC"}
RI_COLOR = {g: ABLATION_COLOR["IMU" if g == "imu" else g] for g in RI_ORDER}

col_idx = {g: group_order.index(g) for g in RI_ORDER}
R_pct = R * 100.0

for g in RI_ORDER:
    v = R_pct[:, col_idx[g]]
    print(f"{RI_LABEL[g].replace(chr(10), ' '):<14} mean={v.mean():5.2f}%  SEM={v.std(ddof=1) / np.sqrt(v.size):.2f}%")


# ── Panel (b) data: policies that learned to walk, per sensor subset ────────
POLICIES_LEARNED = {
    "pos": 0,
    "vel": 5,
    "pos, vel": 0,
    "action": 0,
    "pos, action": 5,
    "vel, action": 5,
    "pos, vel, action": 5,
}
N_POLICIES_TRAINED = 5

# UPSET_LABEL = {"pos": "Position", "vel": "Velocity", "action": "Action",
#                "imu": "IMU", "fc": "Foot Contact"}
UPSET_LABEL = {"pos": "Pos", "vel": "Vel", "action": "Act",
               "imu": "IMU", "fc": "FC"}
UPSET_ROW_ORDER = ["pos", "vel", "action", "imu", "fc"]


# ── Draw ──────────────────────────────────────────────────────────────────────
def draw_relative_importance(ax, R_pct, order):
    """Panel (a): one bar per sensor modality (mean over policies), dots =
    individual policies, whisker = SEM. Mirrors draw_sensor_loss's look."""
    x = np.arange(len(order))
    colors = [RI_COLOR[g] for g in order]
    vals = [R_pct[:, col_idx[g]] for g in order]
    means = np.array([v.mean() for v in vals])
    sems = np.array([v.std(ddof=1) / np.sqrt(v.size) for v in vals])
    width = 0.60
    rng = np.random.default_rng(0)

    ax.bar(x, means, width=width, color=colors, alpha=0.80,
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#222222",
                elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=5)
    for xi, v, c in zip(x, vals, colors):
        jitter = rng.uniform(-width * 0.28, width * 0.28, size=v.size)
        ax.scatter(xi + jitter, v, color='k', alpha=0.55, s=7,
                   linewidths=0, zorder=4)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.1f}%", (xi, m), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([RI_LABEL[g] for g in order], rotation=90)
    ax.set_ylabel("Relative Importance (%)", labelpad=5)
    ax.set_ylim(0, max(means + sems) * 1.30)
    style_axes(ax)


def draw_upset(ax_bars, ax_matrix, learned, complete):
    """Panel (b): bar per sensor subset, dot matrix of which sensors it holds."""
    parse = lambda key: {s.strip() for s in key.split(",")}
    items = sorted(learned.items(), key=lambda kv: (kv[1], len(parse(kv[0]))))
    sets = [parse(k) for k, _ in items]
    values = [v for _, v in items]
    x = np.arange(len(items))

    rows = [s for s in UPSET_ROW_ORDER if s in set().union(*sets)]

    COMPLETE, PARTIAL = "#4169E1", "#9e9e9e"
    ax_bars.bar(x, values, width=0.6, zorder=3,
                color=[COMPLETE if v >= complete else PARTIAL for v in values])
    ax_bars.set_ylabel(f"Policies learned (of {complete})")
    ax_bars.set_ylim(0, complete * 1.18)
    ax_bars.set_xticks([])
    style_axes(ax_bars)

    ACTIVE, INACTIVE = "#b0202f", "#dcdcdc"
    for xi, active in zip(x, sets):
        for r, sensor in enumerate(rows):
            ax_matrix.scatter(xi, r, s=55, zorder=3,
                              color=ACTIVE if sensor in active else INACTIVE)
        on = [r for r, sensor in enumerate(rows) if sensor in active]
        if len(on) >= 2:
            ax_matrix.plot([xi, xi], [min(on), max(on)], color="black", lw=1.6, zorder=2)

    ax_matrix.set_yticks(range(len(rows)))
    ax_matrix.set_yticklabels([UPSET_LABEL[s] for s in rows])
    ax_matrix.set_ylim(len(rows) - 0.5, -0.5)   # first sensor on top
    ax_matrix.set_xlim(-0.5, len(items) - 0.5)
    ax_matrix.set_xticks([])
    style_axes(ax_matrix, grid=None, spines=())
    ax_matrix.tick_params(length=0)


with paper_rc():
    fig = plt.figure(figsize=(PAGE, H_TALL))

    # Both bar axes share one top and one bottom, so their baselines sit at the
    # same level and the two panels stand the same height. What hangs below
    # differs: rotated tick labels on the left, the UpSet matrix on the right.
    BARS_TOP, BARS_BOTTOM = 0.90, 0.38

    gs_a = GridSpec(1, 1, figure=fig, left=0.08, right=0.43,
                    top=BARS_TOP, bottom=BARS_BOTTOM)
    ax_a = fig.add_subplot(gs_a[0])

    gs_b = GridSpec(1, 1, figure=fig, left=0.56, right=0.99,
                    top=BARS_TOP, bottom=BARS_BOTTOM)
    gs_b_matrix = GridSpec(1, 1, figure=fig, left=0.56, right=0.99,
                           top=BARS_BOTTOM - 0.05, bottom=0.04)
    ax_b_bars = fig.add_subplot(gs_b[0])
    ax_b_matrix = fig.add_subplot(gs_b_matrix[0])

    draw_relative_importance(ax_a, R_pct, RI_ORDER)
    draw_upset(ax_b_bars, ax_b_matrix, POLICIES_LEARNED, N_POLICIES_TRAINED)

    panel_label(ax_a, "(a)", x=-0.26, y=1.04)
    panel_label(ax_b_bars, "(b)", x=-0.14, y=1.04)

    save_figure(fig, "relative_importance_upset_combined", jpg=True,pdf=False,png=False)
    plt.show()
