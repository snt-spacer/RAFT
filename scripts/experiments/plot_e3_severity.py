"""Plot E3 severity sweep results.

Compares RAFT (GRU-64-AC, proposed) vs OBS (λ=0, interpretability) across
DEG and STK failure modes at k=1 with varying severity.

Generates docs/figures_old/e2_severity.png  (E2 is now a table in the paper)

Usage:
    python scripts/experiments/plot_e3_severity.py
"""

import json
import statistics
import pathlib
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not found — install it with: pip install matplotlib")
    sys.exit(1)

RESULTS_DIR = pathlib.Path("docs/results/e3_severity_sweep")
SEEDS = [42, 1337, 7]
OUT_PATH = pathlib.Path("docs/figures_old/e2_severity.png")

COLOR_RAFT = "#7C3AED"   # purple — RAFT primary method
COLOR_OBS  = "#F97316"   # orange — OBS interpretability variant


def load_mode(subdir: str):
    """Returns (severities, mean_sr, std_sr, mean_fpe_cm, std_fpe_cm)."""
    per_seed = []
    for seed in SEEDS:
        f = RESULTS_DIR / subdir / f"seed_{seed}" / "severity_results.json"
        if not f.exists():
            print(f"  [SKIP] {subdir}/seed_{seed} — not found")
            continue
        per_seed.append(json.load(open(f)))

    if not per_seed:
        return None, None, None, None, None

    sevs = sorted(per_seed[0].keys(), key=float)
    mean_sr, std_sr, mean_fpe, std_fpe = [], [], [], []
    for s in sevs:
        srs  = [d[s]["success_rate"] * 100 for d in per_seed if s in d]
        fpes = [d[s]["mean_final_pos_m"] * 100 for d in per_seed if s in d]
        mean_sr.append(statistics.mean(srs) if srs else float("nan"))
        std_sr.append(statistics.stdev(srs) if len(srs) > 1 else 0.0)
        mean_fpe.append(statistics.mean(fpes) if fpes else float("nan"))
        std_fpe.append(statistics.stdev(fpes) if len(fpes) > 1 else 0.0)

    return [float(s) for s in sevs], mean_sr, std_sr, mean_fpe, std_fpe


# OBS data (original dirs: deg/, stk/)
obs_deg = load_mode("deg")
obs_stk = load_mode("stk")

# RAFT data (raft/deg/, raft/stk/)
raft_deg = load_mode("raft/deg")
raft_stk = load_mode("raft/stk")


def plot_panel(ax, mode_data_list, title, xlabel):
    """Plot SR (left axis) and FPE (right axis) for multiple methods."""
    ax2 = ax.twinx()
    max_fpe = 0.0

    for (sevs, sr, sr_std, fpe, fpe_std), color, label in mode_data_list:
        if sevs is None:
            continue
        ax.plot(sevs, sr, color=color, lw=2.2, marker="o", ms=5,
                label=f"{label} SR")
        ax.fill_between(sevs,
                        [m - s for m, s in zip(sr, sr_std)],
                        [m + s for m, s in zip(sr, sr_std)],
                        color=color, alpha=0.13)
        ax2.plot(sevs, fpe, color=color, lw=1.8, marker="s", ms=5,
                 linestyle="--", label=f"{label} FPE")
        ax2.fill_between(sevs,
                         [m - s for m, s in zip(fpe, fpe_std)],
                         [m + s for m, s in zip(fpe, fpe_std)],
                         color=color, alpha=0.10)
        if fpe:
            max_fpe = max(max_fpe, max(v for v in fpe if v == v))

    ax.set_ylim(0, 110)
    ax.set_ylabel("Success Rate (%)")
    ax.tick_params(axis="y")

    ax2.set_ylabel("Mean Final Pos Error (cm)")
    ax2.set_ylim(0, max(max_fpe * 2.5, 0.5))
    ax2.tick_params(axis="y")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)

    handles = [
        mpatches.Patch(color=COLOR_RAFT, label="RAFT (solid=SR, dash=FPE)"),
        mpatches.Patch(color=COLOR_OBS,  label="OBS λ=0 (solid=SR, dash=FPE)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(
    "E2 — Failure Severity Sweep: RAFT vs OBS ($\\lambda{=}0$), $k{=}1$, 3 seeds",
    fontsize=13, fontweight="bold"
)

plot_panel(
    axes[0],
    [
        (raft_deg, COLOR_RAFT, "RAFT"),
        (obs_deg,  COLOR_OBS,  "OBS"),
    ],
    title="E2a — Degradation Severity (DEG, k=1)",
    xlabel="Degradation scale  [0 = dead → 1 = healthy]",
)
plot_panel(
    axes[1],
    [
        (raft_stk, COLOR_RAFT, "RAFT"),
        (obs_stk,  COLOR_OBS,  "OBS"),
    ],
    title="E2b — Stuck-On Severity (STK, k=1)",
    xlabel="Stuck offset  [0 = no effect → 1 = fully stuck open]",
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
