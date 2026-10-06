"""Relative sensory-feedback importance (Yu et al., Nat. Mach. Intell. 2023,
arXiv:2306.17101 -- integrated-gradients saliency, grouped by sensor modality
and normalised to sum to 100% per policy: r_o = I_o / sum_o I_o).

Bars = mean r_o across the 10 independently-trained "flat" DAgger students
(integrate_grad/ig/STUDENT-dagger-{0..9}-flat); dots = each policy's own r_o.
Computed by ../../integrate_grad/relative_importance.py -> relative_importance.npy.

Figure geometry/fonts come from paper_style.py, matching the other ICRA figures
in this folder.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import (COL, H_SHORT, ABLATION_COLOR, ABLATION_LABEL,
                          paper_rc, save_figure, style_axes)

ROOT = Path(__file__).resolve().parents[2]
IG_DIR = ROOT / "integrate_grad"

R = np.load(IG_DIR / "relative_importance.npy")  # (num_policies, 5) fractions, sum=1 per row
meta = json.loads((IG_DIR / "relative_importance_meta.json").read_text())
group_order = meta["group_order"]  # ["pos","vel","action","imu","fc"]

# Match the pos/vel/action/fc/IMU ordering used by the sensor-loss ablation figure.
PLOT_ORDER = ["pos", "vel", "action", "fc", "imu"]
key_map = {"imu": "IMU", "pos": "pos", "vel": "vel", "action": "action", "fc": "fc"}
col_idx = {g: group_order.index(g) for g in PLOT_ORDER}

R_pct = R * 100.0
means = np.array([R_pct[:, col_idx[g]].mean() for g in PLOT_ORDER])
sems = np.array([R_pct[:, col_idx[g]].std(ddof=1) / np.sqrt(R_pct.shape[0]) for g in PLOT_ORDER])
colors = [ABLATION_COLOR[key_map[g]] for g in PLOT_ORDER]
labels = [ABLATION_LABEL[key_map[g]].replace("No ", "").replace("\n", " ").strip() or "Foot Contact"
          for g in PLOT_ORDER]
labels = ["Position", "Velocity", "Action", "Foot\nContact", "IMU"]

with paper_rc():
    fig, ax = plt.subplots(figsize=(COL, H_SHORT))

    x = np.arange(len(PLOT_ORDER))
    W = 0.62
    ax.bar(x, means, width=W, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#222222",
                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=5)

    rng = np.random.default_rng(0)
    for i, g in enumerate(PLOT_ORDER):
        vals = R_pct[:, col_idx[g]]
        jitter = rng.uniform(-W * 0.28, W * 0.28, size=vals.size)
        ax.scatter(x[i] + jitter, vals, s=10, color="#0b0b0b", alpha=0.55,
                   linewidths=0, zorder=4)

    for xi, m in zip(x, means):
        ax.annotate(f"{m:.1f}%", (xi, m), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative importance (%)", labelpad=5)
    ax.set_xlabel("Sensor modality", labelpad=6)
    ax.set_ylim(0, max(means + sems) * 1.35)
    ax.set_xlim(-0.6, len(PLOT_ORDER) - 0.4)
    style_axes(ax)
    ax.set_title(
        f"Integrated-gradients relative importance\n"
        f"(bars = mean over {R.shape[0]} policies, dots = individual policies)",
        fontsize=9,
    )

    fig.tight_layout()
    save_figure(fig, "relative_importance_ig")
    plt.show()

print("\nMean relative importance (%):")
for g, m, s in zip(PLOT_ORDER, means, sems):
    print(f"  {g:<8} {m:5.2f} +/- {s:.2f} (SEM)")
