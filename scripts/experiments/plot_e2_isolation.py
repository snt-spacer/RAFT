"""Plot E2 per-mode isolation generalization heatmap.

Generates docs/figures/e2_isolation.png

Usage:
    python scripts/experiments/plot_e2_isolation.py
"""

import json
import statistics
import pathlib
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib/numpy not found — install with: pip install matplotlib numpy")
    sys.exit(1)

BASE   = pathlib.Path("docs/results/e2_per_mode_isolation")
SEEDS  = [42, 1337, 7]
# RAFT first (proposed), VAN-MLP-AC second, then OBS variants
TRAINS = ["RAFT", "VAN_AC", "OBS_ALL", "OBS_DEG", "OBS_DEAD", "OBS_STK"]
EVALS  = ["DEG", "DEAD", "STK"]
OUT    = pathlib.Path("docs/figures/e2_isolation.png")

YLABELS = {
    "RAFT":    "\\textbf{RAFT} (proposed)",
    "VAN_AC":  "VAN-MLP-AC",
    "OBS_ALL": "OBS ($\\lambda{=}0$)",
    "OBS_DEG": "OBS-DEG",
    "OBS_DEAD":"OBS-DEAD",
    "OBS_STK": "OBS-STK",
}

# Build matrix of mean SR at k=2
matrix = np.full((len(TRAINS), len(EVALS)), float("nan"))
matrix_std = np.zeros_like(matrix)

for i, train in enumerate(TRAINS):
    for j, eval_mode in enumerate(EVALS):
        per_seed = []
        for seed in SEEDS:
            f = BASE / train / f"eval_{eval_mode}" / f"seed_{seed}" / "eval_gt_failures.json"
            if not f.exists():
                continue
            d = json.load(open(f))["results"]
            per_seed.append(d["2"]["success_rate"] * 100)
        if per_seed:
            matrix[i, j] = statistics.mean(per_seed)
            matrix_std[i, j] = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.0))
im = ax.imshow(matrix, vmin=80, vmax=100, cmap="Blues")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Success Rate (%) at k=2", fontsize=10)

ax.set_xticks(range(len(EVALS)))
ax.set_yticks(range(len(TRAINS)))
ax.set_xticklabels(["DEG only", "DEAD only", "STK only"], fontsize=11)
ax.set_yticklabels([YLABELS[t] for t in TRAINS], fontsize=11, usetex=False)
ax.set_xlabel("Evaluation mode", fontsize=11)
ax.set_ylabel("Trained on", fontsize=11)
ax.set_title("E2 — Generalisation Matrix (SR\\,\\% at $k{=}2$)", fontsize=12, fontweight="bold")

# Bold border around RAFT row
for j in range(len(EVALS)):
    ax.add_patch(plt.Rectangle(
        (j - 0.5, -0.5), 1, 1,
        fill=False, edgecolor="#7C3AED", lw=2.0, zorder=5
    ))

for i in range(len(TRAINS)):
    for j in range(len(EVALS)):
        val = matrix[i, j]
        if np.isnan(val):
            ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            continue
        txt = f"{val:.0f}\n±{matrix_std[i,j]:.0f}"
        color = "white" if val > 96 else "black"
        weight = "bold" if i == 0 else "normal"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                color=color, fontweight=weight)

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")

