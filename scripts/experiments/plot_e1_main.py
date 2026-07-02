"""Plot E1 main comparison results.

Shows VAN (failure-naive baseline), RAFT (best), OBS / OBS-MSE (interpretability
variants), and Oracle (mixed-mode upper bound, actor sees D_gt).

Generates docs/figures/e1_main_comparison.png

Usage:
    python scripts/experiments/plot_e1_main.py
"""

import json
import statistics
import pathlib
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found — install with: pip install matplotlib")
    sys.exit(1)

E1_BASE  = pathlib.Path("docs/results/e1_main_comparison")
E6_BASE  = pathlib.Path("docs/results/e6_ablation")
E7_BASE  = pathlib.Path("docs/results/e7_gt_collapse")
E11_BASE = pathlib.Path("docs/results/e11_ac_ablation")
SEEDS = [42, 1337, 7]
KS    = [0, 1, 2, 3, 4]
OUT   = pathlib.Path("docs/figures/e1_main_comparison.png")

METHODS = {
    "VAN": {
        "label": "VAN-PPO (failure-naive baseline)",
        "color": "#6B7280", "ls": ":", "marker": "s", "lw": 1.8, "zorder": 2,
        "dirs": [E1_BASE / "VAN" / f"seed_{s}" for s in SEEDS],
    },
    "GT_OBS": {
        "label": "Oracle (actor sees $\\mathbf{D}_{\\mathrm{gt}}$)",
        "color": "#B45309", "ls": "-.", "marker": "P", "lw": 2.0, "zorder": 3,
        "dirs": [E1_BASE / "GT_OBS" / f"seed_{s}" for s in SEEDS],
    },
    "OBS_MSE": {
        "label": "OBS-MSE ($\\lambda{=}1$, interpretability variant)",
        "color": "#6366F1", "ls": "--", "marker": "D", "lw": 1.8, "zorder": 4,
        "dirs": [E6_BASE / "OBS_FULL" / f"seed_{s}" for s in SEEDS],
    },
    "OBS": {
        "label": "OBS ($\\lambda{=}0$, interpretability ext.)",
        "color": "#F97316", "ls": "-.", "marker": "^", "lw": 2.0, "zorder": 5,
        "dirs": [E6_BASE / "ABL_NOMSE" / f"seed_{s}" for s in SEEDS],
    },
    "RAFT": {
        "label": "RAFT — GRU-64-AC",
        "color": "#7C3AED", "ls": "-", "marker": "o", "lw": 2.5, "zorder": 7,
        "dirs": [E11_BASE / "GRU64_AC" / f"seed_{s}" for s in SEEDS],
    },
}


def load_dirs(dirs):
    per_seed = []
    for d in dirs:
        f = pathlib.Path(d) / "eval_gt_failures.json"
        if not f.exists():
            return None, None
        per_seed.append(json.load(open(f))["results"])
    means, stds = [], []
    for k in KS:
        vals = [d[str(k)]["success_rate"] * 100 for d in per_seed if str(k) in d]
        if not vals:
            means.append(float("nan"))
            stds.append(0.0)
            continue
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return means, stds


OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4.5))

for key, cfg in METHODS.items():
    means, stds = load_dirs(cfg["dirs"])
    if means is None:
        print(f"  [SKIP] {key} — results not found")
        continue
    ax.plot(KS, means, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
            marker=cfg["marker"], ms=7, label=cfg["label"], zorder=cfg["zorder"])
    ax.fill_between(KS,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=cfg["color"], alpha=0.12, zorder=cfg["zorder"] - 1)

ax.set_xlabel("Number of failed thrusters ($k$)", fontsize=12)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("E1 — Main Comparison: RAFT vs baselines (mixed failure modes)", fontsize=12, fontweight="bold")
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-2, 108)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
