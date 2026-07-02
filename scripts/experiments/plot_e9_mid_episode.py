"""Plot E4 mid-episode failure injection results.

Compares post-injection SR for RAFT, VAN-MLP-AC, OBS, Oracle, VAN.
Overlays RAFT reset-time SR (E1) as a dashed reference.

Generates docs/figures/e4_mid_episode.png

Usage:
    python scripts/experiments/plot_e9_mid_episode.py
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

E9_BASE  = pathlib.Path("docs/results/e9_mid_episode")
E1_BASE  = pathlib.Path("docs/results/e1_main_comparison")
E11_BASE = pathlib.Path("docs/results/e11_ac_ablation")
SEEDS   = [42, 1337, 7]
KS      = [1, 2, 3, 4]
OUT     = pathlib.Path("docs/figures/e4_mid_episode.png")

VARIANTS = {
    "GT_OBS": {
        "label": "Oracle (instant $D_{gt}$ update)",
        "color": "#B45309", "ls": "-.", "marker": "^",
        "dirs": [E9_BASE / "GT_OBS" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_mid_episode.json",
    },
    "RAFT": {
        "label": "RAFT — GRU-64-AC",
        "color": "#7C3AED", "ls": "-", "marker": "o",
        "dirs": [E9_BASE / "RAFT" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_mid_episode.json",
    },
    "VAN_AC": {
        "label": "VAN-MLP-AC (no recurrence)",
        "color": "#0EA5E9", "ls": "-", "marker": "P",
        "dirs": [E9_BASE / "VAN_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_mid_episode.json",
    },
    "OBS": {
        "label": "OBS ($\\lambda{=}0$) — mid-episode (interp. ext.)",
        "color": "#F97316", "ls": "--", "marker": "D",
        "dirs": [E9_BASE / "OBS" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_mid_episode.json",
    },
    "VAN": {
        "label": "VAN — mid-episode (no failure info)",
        "color": "#6B7280", "ls": ":", "marker": "s",
        "dirs": [E9_BASE / "VAN" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_mid_episode.json",
    },
    "RAFT_RESET": {
        "label": "RAFT — reset-time (E1 reference)",
        "color": "#C4B5FD", "ls": "--", "marker": "o",
        "dirs": [E11_BASE / "GRU64_AC" / f"seed_{s}" for s in SEEDS],
        "fname": "eval_gt_failures.json",
    },
}


def load_dirs(dirs, fname, ks):
    per_seed = []
    for d in dirs:
        f = pathlib.Path(d) / fname
        if not f.exists():
            return None, None
        per_seed.append(json.load(open(f))["results"])
    means, stds = [], []
    for k in ks:
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
    means, stds = load_dirs(cfg["dirs"], cfg["fname"], KS)
    if means is None:
        print(f"  [SKIP] {key} — results not found")
        continue
    lw = 2.0 if key == "RAFT_RESET" else 2.5
    alpha = 0.06 if key == "RAFT_RESET" else 0.12
    ax.plot(KS, means, color=cfg["color"], lw=lw, ls=cfg["ls"],
            marker=cfg["marker"], ms=7, label=cfg["label"], zorder=3)
    ax.fill_between(KS,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=cfg["color"], alpha=alpha, zorder=2)

ax.axvline(x=0, color="black", lw=0)  # invisible, just for layout
ax.set_xlabel("Number of failed thrusters ($k$)", fontsize=12)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("E4 — Mid-Episode Failure Injection: RAFT vs baselines (inject at step 100)", fontsize=12, fontweight="bold")
ax.set_xlim(0.8, 4.2)
ax.set_ylim(-2, 108)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
