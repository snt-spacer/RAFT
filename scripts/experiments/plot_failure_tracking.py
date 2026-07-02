"""Plot E11 — Failure Tracking / Observer Explainability.

Three complementary figures showing how well each method predicts D_gt:

  Figure 1 (failure_readout.png):
    Thruster-by-thruster snapshot at step 200 of a single episode.
    4×3 grid (k=1..4 × DEG/DEAD/STK). Each cell shows 8 thrusters as
    coloured bars — GT (true state), OBS-MSE, OBS, and RAFT probe.

  Figure 2 (prediction_scatter.png):
    Aggregate scatter D_gt vs D_hat over many episodes.
    3×3 grid (methods × modes). Colour by k. R² annotation.

  Figure 3 (adaptation_trajectory.png):
    Single-episode time series for k=2, all 3 modes.
    Shows how D_hat evolves over the 400-step episode.
    RAFT probe computed via linear regression on aggregate data.

Usage:
    python scripts/experiments/plot_failure_tracking.py
"""

import pathlib
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    print("matplotlib not found — pip install matplotlib")
    sys.exit(1)

BASE   = pathlib.Path("docs/results/e11_failure_tracking")
OUT    = pathlib.Path("docs/figures")
MODES  = ["DEG", "DEAD", "STK"]
KS     = [1, 2, 3, 4]
SEED   = 42
N_THR  = 8   # thrusters
D_GT   = 16  # 8 scales + 8 offsets

# ── Colour scheme (matches paper) ─────────────────────────────────────────────
COLORS = {
    "OBS_MSE": "#B45309",   # amber
    "OBS":     "#6B7280",   # gray
    "RAFT":    "#7C3AED",   # purple
    "GT":      "#111827",   # near-black
}
LABELS = {
    "OBS_MSE": "OBS-MSE ($\\lambda{=}1$)",
    "OBS":     "OBS ($\\lambda{=}0$)",
    "RAFT":    "RAFT probe",
    "GT":      "Ground Truth $D_{gt}$",
}

METHODS = ["OBS_MSE", "OBS", "RAFT"]
OUT.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_npz(method, mode, k):
    f = BASE / method / f"seed_{SEED}" / f"{mode}_k{k}.npz"
    if not f.exists():
        return None
    return dict(np.load(f))


def load_all():
    """Return nested dict data[method][mode][k] = npz-dict."""
    data = {}
    for m in METHODS:
        data[m] = {}
        for mode in MODES:
            data[m][mode] = {}
            for k in KS:
                d = load_npz(m, mode, k)
                data[m][mode][k] = d
    return data


# ── RAFT linear probe (fit on aggregate data across all modes+k) ──────────────

def fit_raft_probe(data):
    """Fit ridge regression: h (hidden, N×H) → D_gt (N×16). Returns W (H×16)."""
    H_list, Y_list = [], []
    for mode in MODES:
        for k in KS:
            d = data["RAFT"][mode][k]
            if d is None:
                continue
            H_list.append(d["agg_pred"])   # (M, hidden_dim)
            Y_list.append(d["agg_dgt"])    # (M, 16)
    if not H_list:
        return None
    H = np.concatenate(H_list, axis=0)   # (N, H)
    Y = np.concatenate(Y_list, axis=0)   # (N, 16)
    # Ridge: W = (H'H + λI)⁻¹ H'Y
    lam = 1e-3
    W = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ Y)
    r2_global = r2_score(Y, H @ W)
    print(f"  [RAFT probe] R² (global, all modes+k) = {r2_global:.4f}")
    return W


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def apply_probe(h, W):
    """Apply linear probe: h (N, H) × W (H, 16) → pred (N, 16), clipped [0,1]."""
    return np.clip(h @ W, 0.0, 1.0)


# ── Helper: get predicted scale for a single episode ─────────────────────────

def effective_value(arr16):
    """Effective thrust = clip(scale + offset, 0, 1) — distinguishes STK from DEAD.

    DEAD:    scale=0, offset=0  → effective=0 (dark red)
    STK 0.7: scale=0, offset=0.7 → effective=0.7 (yellow-green)
    DEG 0.3: scale=0.3, offset=0 → effective=0.3 (orange)
    Healthy: scale=1, offset=0  → effective=1.0 (bright green)
    """
    scale  = arr16[..., :N_THR]
    offset = arr16[..., N_THR:]
    return np.clip(scale + offset, 0.0, 1.0)


def get_single_eff(data, method, mode, k, probe_W=None):
    """Return (T, 8) effective-thrust prediction for env[0]'s single episode."""
    d = data[method][mode][k]
    if d is None:
        return None
    pred = d["single_pred"]   # (T, D)
    if method == "RAFT" and probe_W is not None:
        pred = apply_probe(pred, probe_W)  # (T, 16)
    if pred.shape[-1] < 2 * N_THR:
        return pred[:, :N_THR]   # fallback: scale only
    return effective_value(pred)   # (T, 8)


def get_single_gt_eff(data, mode, k):
    """Return (T, 8) GT effective thrust for env[0]."""
    for m in METHODS:
        d = data[m][mode][k]
        if d is not None:
            return effective_value(d["single_dgt"])   # (T, 8)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Failure Readout Map
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_STEP = 200   # representative mid-episode step


def make_figure1(data, probe_W):
    """4×3 grid: snapshot of thruster-by-thruster predictions at step 200."""
    n_rows, n_cols = len(KS), len(MODES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 2.8),
                             squeeze=False)

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.0, vmax=1.0)

    method_rows = ["GT", "OBS_MSE", "OBS", "RAFT"]
    row_labels  = [LABELS["GT"], LABELS["OBS_MSE"], LABELS["OBS"], LABELS["RAFT"]]
    row_colors  = ["#111827", COLORS["OBS_MSE"], COLORS["OBS"], COLORS["RAFT"]]

    for ri, k in enumerate(KS):
        for ci, mode in enumerate(MODES):
            ax = axes[ri][ci]

            # Build (n_method_rows, 8) matrix of EFFECTIVE THRUST
            # = clip(scale + offset, 0, 1): DEAD→0, STK 0.7→0.7, DEG 0.3→0.3
            mat = np.full((len(method_rows), N_THR), np.nan)

            # GT row
            gt_eff = get_single_gt_eff(data, mode, k)
            if gt_eff is not None:
                mat[0] = gt_eff[SNAPSHOT_STEP]

            # OBS_MSE
            pred = get_single_eff(data, "OBS_MSE", mode, k)
            if pred is not None:
                mat[1] = pred[SNAPSHOT_STEP]

            # OBS
            pred = get_single_eff(data, "OBS", mode, k)
            if pred is not None:
                mat[2] = pred[SNAPSHOT_STEP]

            # RAFT probe
            pred = get_single_eff(data, "RAFT", mode, k, probe_W)
            if pred is not None:
                mat[3] = pred[SNAPSHOT_STEP]

            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                           interpolation="nearest")

            # Annotate cells with value
            for r in range(len(method_rows)):
                for c in range(N_THR):
                    val = mat[r, c]
                    txt_color = "white" if val < 0.45 or val > 0.88 else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color=txt_color)

            # Y-axis labels (only leftmost column)
            if ci == 0:
                ax.set_yticks(range(len(method_rows)))
                ax.set_yticklabels(row_labels, fontsize=7.5)
                # Bold GT row
                for lbl, clr in zip(ax.get_yticklabels(), row_colors):
                    lbl.set_color(clr)
            else:
                ax.set_yticks([])

            # X-axis labels (only bottom row)
            if ri == n_rows - 1:
                ax.set_xticks(range(N_THR))
                ax.set_xticklabels([f"T{i}" for i in range(N_THR)], fontsize=8)
            else:
                ax.set_xticks([])

            # Title for top row
            if ri == 0:
                ax.set_title(f"Mode: {mode}", fontsize=10, fontweight="bold")

            # K label on left
            if ci == 0:
                ax.set_ylabel(f"k={k}", fontsize=9, rotation=0, labelpad=30,
                              va="center")

            # Draw thick border around GT row
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.axhline(0.5, color="white", lw=1.5)  # separator GT / predictions

    # Shared colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, shrink=0.6)
    cbar.set_label("Scale value (0 = failed, 1 = healthy)", fontsize=9)

    fig.suptitle(
        f"E9 — Failure Readout Map  (snapshot at step {SNAPSHOT_STEP})  "
        r"• Color = effective thrust = $\min(s_i + \delta_i, 1)$" + "\n"
        "DEAD→0 (red), STK→offset (yellow), DEG→scale (orange), Healthy→1 (green)  "
        "| Rows: GT | OBS-MSE | OBS (λ=0) | RAFT probe",
        fontsize=9.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = OUT / "e9_failure_readout.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Aggregate Scatter: D_gt vs D_hat
# ─────────────────────────────────────────────────────────────────────────────

K_COLORS = {1: "#A78BFA", 2: "#3B82F6", 3: "#F97316", 4: "#DC2626"}


def make_figure2(data, probe_W):
    """3×3 scatter: methods × modes. Each cell: D_gt_scale vs D_hat_scale."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), squeeze=False)

    for ri, method in enumerate(METHODS):
        for ci, mode in enumerate(MODES):
            ax = axes[ri][ci]

            r2_vals = []
            for k in KS:
                d = data[method][mode][k]
                if d is None:
                    continue
                dgt_eff = effective_value(d["agg_dgt"]).ravel()   # (M*8,)
                pred = d["agg_pred"]
                if method == "RAFT" and probe_W is not None:
                    pred = apply_probe(pred, probe_W)
                if pred.shape[-1] >= 2 * N_THR:
                    pred_eff = effective_value(pred).ravel()
                else:
                    pred_eff = pred[:, :N_THR].ravel()

                ax.scatter(dgt_eff, pred_eff, s=0.5, alpha=0.3,
                           color=K_COLORS[k], rasterized=True)
                r2_vals.append(r2_score(dgt_eff, pred_eff))

            # Perfect prediction diagonal
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.grid(True, ls=":", alpha=0.4)

            if r2_vals:
                r2_mean = np.mean(r2_vals)
                ax.text(0.03, 0.93, f"R²={r2_mean:.3f}", transform=ax.transAxes,
                        fontsize=9, color=COLORS[method], fontweight="bold",
                        va="top")

            if ri == 0:
                ax.set_title(f"{mode}", fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(LABELS[method] + "\n$\\hat{D}_{eff}$", fontsize=8.5,
                              color=COLORS[method])
            if ri == 2:
                ax.set_xlabel("True $D_{gt,eff}$ (scale + offset)", fontsize=9)

    # Legend for k colours
    handles = [mpatches.Patch(color=K_COLORS[k], label=f"k={k}") for k in KS]
    fig.legend(handles=handles, loc="upper right", fontsize=9,
               bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    fig.suptitle(
        "E9 — Aggregate Prediction Accuracy: $D_{gt}$ vs $\\hat{D}$ (scale dim)\n"
        "R² computed per mode across k=1..4",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out_path = OUT / "e9_prediction_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Single-episode adaptation trajectory
# ─────────────────────────────────────────────────────────────────────────────

def make_figure3(data, probe_W, k_traj=4):
    """k=4 × 3 modes: time-series of mean effective thrust over failed thrusters."""
    k = k_traj
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, squeeze=False)

    for ci, mode in enumerate(MODES):
        ax = axes[0][ci]

        # Ground truth: use OBS_MSE D_gt (same for all methods)
        d_ref = data["OBS_MSE"][mode][k]
        if d_ref is None:
            ax.set_title(f"{mode}\n(no data)", fontsize=10)
            continue

        dgt_all = d_ref["single_dgt"]   # (T, 16)
        dgt_eff = effective_value(dgt_all)   # (T, 8)
        T = dgt_eff.shape[0]
        steps = np.arange(T)

        # Identify failed thrusters: effective value < 0.95 on average
        mean_eff_thr = dgt_eff.mean(axis=0)   # (8,)
        failed_thr = mean_eff_thr < 0.95
        if not failed_thr.any():
            failed_thr = mean_eff_thr < mean_eff_thr.max() * 0.99

        def mean_failed(eff_mat):
            cols = eff_mat[:, failed_thr]
            return cols.mean(axis=1) if cols.shape[1] > 0 else eff_mat.mean(axis=1)

        gt_line = mean_failed(dgt_eff)
        ax.fill_between(steps, 0, gt_line, alpha=0.08, color=COLORS["GT"])
        ax.step(steps, gt_line, color=COLORS["GT"], lw=1.5, ls="--",
                label=LABELS["GT"], where="post")

        # Method predictions
        for method in METHODS:
            d = data[method][mode][k]
            if d is None:
                continue
            pred = d["single_pred"]   # (T, D)
            if method == "RAFT" and probe_W is not None:
                pred = apply_probe(pred, probe_W)   # (T, 16)
            if pred.shape[-1] < 2 * N_THR:
                pred_eff = pred[:, :N_THR]   # fallback: scale only
            else:
                pred_eff = effective_value(pred)   # (T, 8)
            pred_line = mean_failed(pred_eff)

            # Smooth slightly for readability (rolling mean window=5)
            from numpy.lib.stride_tricks import sliding_window_view
            w = 5
            pad = w // 2
            padded = np.pad(pred_line, (pad, pad), mode="edge")
            smoothed = sliding_window_view(padded, w).mean(axis=-1)[:T]

            # Reset markers (where env[0] episode ended)
            reset_steps_bool = d.get("reset_steps", np.zeros(T, dtype=bool))
            reset_idx = np.where(reset_steps_bool)[0]

            ls = "-" if method == "OBS_MSE" else ("--" if method == "OBS" else "-.")
            lw = 2.0 if method == "OBS_MSE" else 1.5
            ax.plot(steps, smoothed, color=COLORS[method], lw=lw, ls=ls,
                    label=LABELS[method], alpha=0.9)

            # Mark episode resets with vertical dotted lines
            for rs in reset_idx:
                ax.axvline(rs, color=COLORS[method], lw=0.6, ls=":", alpha=0.4)

        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(gt_line.mean(), color=COLORS["GT"], lw=0.6, ls=":", alpha=0.5)
        ax.set_title(f"Mode: {mode}  (k={k})", fontsize=10, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.4)
        if ci == 0:
            ax.set_ylabel("Mean effective thrust of affected thrusters\n(scale + offset, 0=failed, 1=healthy)", fontsize=9)
        if ci == 2:
            ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"E9 — Single-Episode Trajectory (k={k}, env[0])  •  "
        "Effective thrust: DEAD→0, STK→offset, DEG→scale\n"
        "OBS-MSE (supervised) converges to $D_{gt}$ over ~50–100 steps. "
        "OBS ($\\lambda{=}0$) outputs a constant ~0.5. RAFT probe extracts minimal failure info.",
        fontsize=9.5, fontweight="bold",
    )
    plt.tight_layout()
    out_path = OUT / "e9_adaptation_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_all()

    # Check which methods have data
    for m in METHODS:
        found = sum(
            1 for mode in MODES for k in KS
            if data[m][mode][k] is not None
        )
        print(f"  {m}: {found}/12 scenario files found")

    # Fit RAFT linear probe
    probe_W = None
    if any(data["RAFT"][mode][k] is not None for mode in MODES for k in KS):
        print("\nFitting RAFT linear probe...")
        probe_W = fit_raft_probe(data)
        if probe_W is not None:
            print(f"  Probe shape: {probe_W.shape}")

    print("\nGenerating Figure 1 (failure readout map)...")
    make_figure1(data, probe_W)

    print("Generating Figure 2 (prediction scatter)...")
    make_figure2(data, probe_W)

    print("Generating Figure 3 (adaptation trajectory)...")
    make_figure3(data, probe_W)

    print("\nDone.")


if __name__ == "__main__":
    main()
