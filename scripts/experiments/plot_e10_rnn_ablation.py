"""Plot E5 recurrent policy ablation results.

Compares VAN-based recurrent variants (GRU/LSTM × 64/256) against VAN-MLP — all
trained WITHOUT the asymmetric critic, to test whether recurrence alone can
detect failures.

Generates docs/figures/e5_rnn_ablation.png

Usage:
    python scripts/experiments/plot_e10_rnn_ablation.py
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

E10_BASE = pathlib.Path("docs/results/e10_rnn_ablation")
E1_BASE  = pathlib.Path("docs/results/e1_main_comparison")
E6_BASE  = pathlib.Path("docs/results/e6_ablation")
SEEDS    = [42, 1337, 7]
KS       = [0, 1, 2, 3, 4]
OUT      = pathlib.Path("docs/figures/e5_rnn_ablation.png")

VARIANTS = {
    "GRU64": {
        "label": "VAN-GRU-64",
        "color": "#DC2626", "ls": "-", "marker": "^", "lw": 2.2, "zorder": 4,
        "dirs": [E10_BASE / "GRU64" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "GRU256": {
        "label": "VAN-GRU-256",
        "color": "#F97316", "ls": "-.", "marker": "D", "lw": 2.0, "zorder": 3,
        "dirs": [E10_BASE / "GRU256" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "LSTM64": {
        "label": "VAN-LSTM-64",
        "color": "#7C3AED", "ls": "--", "marker": "s", "lw": 2.0, "zorder": 3,
        "dirs": [E10_BASE / "LSTM64" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "LSTM256": {
        "label": "VAN-LSTM-256",
        "color": "#BE185D", "ls": (0, (5, 2, 1, 2)), "marker": "P", "lw": 2.0, "zorder": 3,
        "dirs": [E10_BASE / "LSTM256" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
    "VAN": {
        "label": "VAN-MLP (no history)",
        "color": "#374151", "ls": ":", "marker": "x", "lw": 2.0, "zorder": 5,
        "dirs": [E1_BASE / "VAN" / f"seed_{s}" for s in SEEDS],
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
ax.set_title("E5 — Recurrent Policy Ablation: GRU / LSTM vs VAN-MLP (no asymmetric critic)", fontsize=12, fontweight="bold")
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-2, 108)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
