"""Plot E6 asymmetric critic ablation results.

Compares five AC variants — VAN-MLP-AC and VAN-{GRU,LSTM}-{64,256}-AC — all
sharing the same privileged asymmetric critic. RAFT is GRU-64-AC.

Generates docs/figures/e6_ac_ablation.png

Usage:
    python scripts/experiments/plot_e11_ac_ablation.py
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

E11_BASE = pathlib.Path("docs/results/e11_ac_ablation")
SEEDS    = [42, 1337, 7]
KS       = [0, 1, 2, 3, 4]
OUT      = pathlib.Path("docs/figures/e6_ac_ablation.png")

VARIANTS = {
    "GRU64_AC": {
        "label": "VAN-GRU-64-AC (RAFT)",
        "color": "#7C3AED", "ls": "-", "marker": "o", "lw": 2.5, "zorder": 5,
        "dirs": [E11_BASE / "GRU64_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "LSTM64_AC": {
        "label": "VAN-LSTM-64-AC",
        "color": "#1D4ED8", "ls": "--", "marker": "s", "lw": 2.0, "zorder": 4,
        "dirs": [E11_BASE / "LSTM64_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "VAN_AC": {
        "label": "VAN-MLP-AC (memoryless)",
        "color": "#DC2626", "ls": "-.", "marker": "^", "lw": 2.0, "zorder": 4,
        "dirs": [E11_BASE / "VAN_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "GRU256_AC": {
        "label": "VAN-GRU-256-AC",
        "color": "#F97316", "ls": (0, (5, 2)), "marker": "D", "lw": 2.0, "zorder": 3,
        "dirs": [E11_BASE / "GRU256_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "LSTM256_AC": {
        "label": "VAN-LSTM-256-AC",
        "color": "#BE185D", "ls": (0, (5, 2, 1, 2)), "marker": "v", "lw": 2.0, "zorder": 3,
        "dirs": [E11_BASE / "LSTM256_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
}


def load_dirs(dirs, fname):
    per_seed = []
    for d in dirs:
        f = pathlib.Path(d) / fname
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
fig, ax = plt.subplots(figsize=(8, 5))

for key, cfg in VARIANTS.items():
    means, stds = load_dirs(cfg["dirs"], cfg["fname"])
    if means is None:
        print(f"  [SKIP] {key} — results not found")
        continue
    ax.plot(KS, means, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
            marker=cfg["marker"], ms=7, label=cfg["label"], zorder=cfg["zorder"])
    ax.fill_between(KS,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=cfg["color"], alpha=0.10, zorder=cfg["zorder"] - 1)

ax.set_xlabel("Number of failed thrusters ($k$)", fontsize=12)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("E6 — Asymmetric Critic Ablation: actor architecture under shared privileged critic", fontsize=12, fontweight="bold")
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-2, 108)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
