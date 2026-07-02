"""Plot E8 observer interpretability probe results.

Two panels:
  Left:  Linear probe R² at k=0 and k=2 for OBS_FULL vs ABL_NOMSE.
  Right: Zeroing test ΔSR (normal − zeroed) vs k for both variants.

Generates docs/figures/e8_interpretability.png

Usage:
    python scripts/experiments/plot_e8_interpretability.py
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
    print("matplotlib/numpy not found")
    sys.exit(1)

BASE  = pathlib.Path("docs/results/e8_observer_probe")
SEEDS = [42, 1337, 7]
KS    = [0, 1, 2, 3, 4]
OUT   = pathlib.Path("docs/figures/e5_interpretability.png")

VARIANTS = {
    "OBS_FULL":  {"label": "OBS-FULL",  "color": "#1D4ED8"},
    "ABL_NOMSE": {"label": "ABL-NOMSE", "color": "#DC2626"},
}


def load_probes(variant):
    results = []
    for seed in SEEDS:
        f = BASE / variant / f"seed_{seed}" / "probe_results.json"
        if f.exists():
            results.append(json.load(open(f)))
    return results


OUT.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left panel: Linear probe R² ───────────────────────────────────────────────
ax = axes[0]
x = np.arange(2)  # k=0, k=2
width = 0.35
offsets = [-width/2, width/2]

for i, (variant, cfg) in enumerate(VARIANTS.items()):
    data = load_probes(variant)
    if not data:
        print(f"  [SKIP] {variant} probe results not found")
        continue

    r2_k0 = [d["linear_probe_k0"]["r2_overall"] for d in data]
    r2_k2 = [d["linear_probe_k2"]["r2_overall"] for d in data]

    means = [statistics.mean(r2_k0), statistics.mean(r2_k2)]
    errs = [
        statistics.stdev(r2_k0) if len(r2_k0) > 1 else 0,
        statistics.stdev(r2_k2) if len(r2_k2) > 1 else 0,
    ]

    ax.bar(x + offsets[i], means, width, label=cfg["label"],
           color=cfg["color"], alpha=0.85, yerr=errs, capsize=4, error_kw={"lw": 1.5})

ax.set_xlabel("Failure count at collection time", fontsize=11)
ax.set_ylabel("R² (linear probe D_hat → D_gt)", fontsize=11)
ax.set_title("Linear Probe: Is D_hat predictive of D_gt?", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["k=0\n(no failures)", "k=2"])
ax.set_ylim(-0.05, 1.05)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=10)

# ── Right panel: Zeroing test ΔSR ─────────────────────────────────────────────
ax = axes[1]
for variant, cfg in VARIANTS.items():
    data = load_probes(variant)
    if not data:
        continue

    delta_means, delta_stds = [], []
    for k in KS:
        deltas = [
            d["zeroing_test"][str(k)]["delta_sr"]
            for d in data if "zeroing_test" in d and str(k) in d["zeroing_test"]
        ]
        delta_means.append(statistics.mean(deltas) * 100 if deltas else float("nan"))
        delta_stds.append(statistics.stdev(deltas) * 100 if len(deltas) > 1 else 0)

    ax.plot(KS, delta_means, color=cfg["color"], lw=2.5, marker="o", ms=7, label=cfg["label"])
    ax.fill_between(KS,
                    [m - s for m, s in zip(delta_means, delta_stds)],
                    [m + s for m, s in zip(delta_means, delta_stds)],
                    color=cfg["color"], alpha=0.12)

ax.axhline(0, color="black", lw=1.0, ls="--", label="no effect")
ax.set_xlabel("Number of failed thrusters ($k$)", fontsize=11)
ax.set_ylabel("ΔSR = SR_normal − SR_zeroed (pp)", fontsize=11)
ax.set_title("Zeroing Test: Policy use of D_hat", fontsize=11, fontweight="bold")
ax.set_xlim(-0.2, 4.2)
ax.set_xticks(KS)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=10)

plt.suptitle("E5 — Observer Interpretability", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
