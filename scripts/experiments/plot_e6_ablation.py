"""Plot E6 ablation study results (Observer extension ablation).

RAFT (proposed method) is shown as the top reference line.
Observer extension variants are compared against each other.

Generates docs/figures/e4_ablation.png

Usage:
    python scripts/experiments/plot_e6_ablation.py
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

E6_BASE  = pathlib.Path("docs/results/e6_ablation")
E11_BASE = pathlib.Path("docs/results/e11_ac_ablation")
SEEDS    = [42, 1337, 7]
KS       = [0, 1, 2, 3, 4]
OUT      = pathlib.Path("docs/figures/e4_ablation.png")

OBS_VARIANTS = {
    "ABL_NOMSE":    ("OBS ($\\lambda{=}0$, interp. ext.)",        "#F97316", "-",           "o"),
    "ABL_NODETACH": ("ABL-NODETACH",                              "#B45309", "-.",           "P"),
    "OBS_FULL":     ("OBS-MSE ($\\lambda{=}1$, supervised)",      "#6366F1", "--",           "D"),
    "ABL_HIST16":   ("ABL-HIST16",                                "#16A34A", "-.",           "^"),
    "ABL_HIST128":  ("ABL-HIST128",                               "#9333EA", ":",            "v"),
    "ABL_SMALLOBS": ("ABL-SMALLOBS",                              "#EA580C", (0,(3,1,1,1)), "s"),
}


def load_e6(variant):
    per_seed = []
    for seed in SEEDS:
        f = E6_BASE / variant / f"seed_{seed}" / "eval_gt_failures.json"
        if not f.exists():
            return None, None
        per_seed.append(json.load(open(f))["results"])
    means, stds = [], []
    for k in KS:
        vals = [d[str(k)]["success_rate"] * 100 for d in per_seed]
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return means, stds


def load_raft():
    per_seed = []
    for seed in SEEDS:
        f = E11_BASE / "GRU64_AC" / f"seed_{seed}" / "eval_gt_failures.json"
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

# RAFT as top reference
raft_means, raft_stds = load_raft()
if raft_means is not None:
    ax.plot(KS, raft_means, color="#7C3AED", lw=2.5, ls="-", marker="o", ms=7,
            label="RAFT (proposed, reference)", zorder=6)
    ax.fill_between(KS,
                    [m - s for m, s in zip(raft_means, raft_stds)],
                    [m + s for m, s in zip(raft_means, raft_stds)],
                    color="#7C3AED", alpha=0.14, zorder=5)

# Observer extension variants
lws = {"ABL_NOMSE": 2.2}
for variant, (label, color, ls, marker) in OBS_VARIANTS.items():
    means, stds = load_e6(variant)
    if means is None:
        print(f"  [SKIP] {variant}")
        continue
    lw = lws.get(variant, 1.8)
    ax.plot(KS, means, color=color, lw=lw, ls=ls, marker=marker, ms=6, label=label)
    ax.fill_between(KS,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.10)

ax.set_xlabel("Number of failed thrusters ($k$)", fontsize=11)
ax.set_ylabel("Success Rate (%)", fontsize=11)
ax.set_title("E4 — Observer Extension Ablation vs RAFT (proposed)", fontsize=12, fontweight="bold")
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-2, 108)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
