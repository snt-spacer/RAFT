"""Plot E7 GT collapse investigation results.

Compares GT-ORIGINAL, GT-LONGTR, GT-NOCURR, and OBS-FULL.

Generates docs/figures/e7_gt_collapse.png

Usage:
    python scripts/experiments/plot_e7_gt_collapse.py
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
E7_BASE  = pathlib.Path("docs/results/e7_gt_collapse")
SEEDS    = [42, 1337, 7]
KS       = [0, 1, 2, 3, 4]
OUT      = pathlib.Path("docs/figures/e7_gt_collapse.png")

VARIANTS = {
    "GT-ORIGINAL": {
        "label": "GT-ORIGINAL (2000 iters, k_max≤2 at end)",
        "color": "#6B7280", "ls": ":", "marker": "s",
        "dirs": [E1_BASE / "GT" / f"seed_{s}" for s in SEEDS],
    },
    "GT-LONGTR": {
        "label": "GT-LONGTR (5000 iters, curriculum completes)",
        "color": "#B45309", "ls": "--", "marker": "^",
        "dirs": [E7_BASE / "GT_LONGTR" / f"seed_{s}" for s in SEEDS],
    },
    "GT-NOCURR": {
        "label": "GT-NOCURR (2000 iters, k=4 from step 1)",
        "color": "#DC2626", "ls": "-.", "marker": "P",
        "dirs": [E7_BASE / "GT_NOCURR" / f"seed_{s}" for s in SEEDS],
    },
    "OBS-FULL": {
        "label": "OBS-FULL (reference)",
        "color": "#1D4ED8", "ls": "-", "marker": "o",
        "dirs": [E1_BASE / "OBS" / f"seed_{s}" for s in SEEDS],
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
        vals = [d[str(k)]["success_rate"] * 100 for d in per_seed]
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return means, stds


OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))

for key, cfg in VARIANTS.items():
    means, stds = load_dirs(cfg["dirs"])
    if means is None:
        print(f"  [SKIP] {key} — results not found")
        continue
    ax.plot(KS, means, color=cfg["color"], lw=2.5, ls=cfg["ls"],
            marker=cfg["marker"], ms=7, label=cfg["label"], zorder=3)
    ax.fill_between(KS,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=cfg["color"], alpha=0.10, zorder=2)

ax.set_xlabel("Number of failed thrusters ($k$)", fontsize=12)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("E7 — GT Collapse Investigation", fontsize=12, fontweight="bold")
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-2, 108)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
