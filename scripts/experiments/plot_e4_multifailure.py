"""Plot E3 multi-failure scalability results.

Compares RAFT (GRU-64-AC), OBS (λ=0), and the mixed-mode Oracle
across DEG/DEAD/STK modes. Oracle is trained on the same 3-mode failure
curriculum as RAFT/OBS with the actor seeing D_gt, so it is evaluable per mode.

Generates docs/figures/e3_multifailure.png

Usage:
    python scripts/experiments/plot_e4_multifailure.py
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
    print("matplotlib not found — install it with: pip install matplotlib")
    sys.exit(1)

BASE   = pathlib.Path("docs/results/e4_multifailure_scalability")
SEEDS  = [42, 1337, 7]
KS     = [0, 1, 2, 3, 4]
MODES  = ["DEG", "DEAD", "STK"]
OUT    = pathlib.Path("docs/figures/e3_multifailure.png")


def load(method: str, mode: str):
    per_seed = []
    for seed in SEEDS:
        f = BASE / method / mode / f"seed_{seed}" / "eval_gt_failures.json"
        if not f.exists():
            print(f"  [SKIP] {method}/{mode}/seed_{seed} — not found")
            continue
        per_seed.append(json.load(open(f))["results"])
    if not per_seed:
        return None, None
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


# Oracle is loaded per-mode (mixed-mode trained, evaluated under mode pinning).
# Each panel shows the GT result for its own mode.
def _load_gt(mode):
    means, stds = load("GT", mode)
    if means is None:
        print(f"  [WARN] Oracle results not found for {mode}")
        return [float("nan")] * len(KS), [0.0] * len(KS)
    return means, stds

MODE_COLORS = {"DEG": "#2563EB", "DEAD": "#DC2626", "STK": "#16A34A"}
MODE_LABELS = {"DEG": "DEG (continuous)", "DEAD": "DEAD (binary)", "STK": "STK (stuck-on)"}

COLOR_RAFT = "#7C3AED"   # purple — primary proposed method
COLOR_OBS  = "#F97316"   # orange — OBS interpretability variant

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
fig.suptitle(
    "E3 — Per-Mode Scalability: RAFT vs OBS ($\\lambda{=}0$) vs Oracle",
    fontsize=13, fontweight="bold"
)

for ax, mode in zip(axes, MODES):
    raft_means, raft_stds = load("RAFT", mode)
    obs_means,  obs_stds  = load("OBS",  mode)
    gt_means,   gt_stds   = _load_gt(mode)

    # Oracle (mixed-mode oracle, evaluated per mode)
    ax.plot(KS, gt_means, color="grey", lw=2, ls="--", marker="^", ms=6,
            label="Oracle")
    ax.fill_between(KS,
                    [m - s for m, s in zip(gt_means, gt_stds)],
                    [m + s for m, s in zip(gt_means, gt_stds)],
                    color="grey", alpha=0.12)

    # OBS (interpretability variant)
    if obs_means is not None:
        ax.plot(KS, obs_means, color=COLOR_OBS, lw=2, ls="-.", marker="s", ms=5,
                label="OBS ($\\lambda{=}0$, interp. ext.)", alpha=0.85)
        ax.fill_between(KS,
                        [m - s for m, s in zip(obs_means, obs_stds)],
                        [m + s for m, s in zip(obs_means, obs_stds)],
                        color=COLOR_OBS, alpha=0.12)

    # RAFT
    if raft_means is not None:
        ax.plot(KS, raft_means, color=COLOR_RAFT, lw=2.5, marker="o", ms=6,
                label="RAFT")
        ax.fill_between(KS,
                        [m - s for m, s in zip(raft_means, raft_stds)],
                        [m + s for m, s in zip(raft_means, raft_stds)],
                        color=COLOR_RAFT, alpha=0.18)

    ax.set_title(MODE_LABELS[mode])
    ax.set_xlabel("Number of failed thrusters (k)")
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-2, 108)
    ax.set_xticks(KS)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Success Rate (%)")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
