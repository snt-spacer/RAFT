#!/usr/bin/env bash
# ============================================================
# Failure Tracking Evaluation (E11 explainability ablation)
#
# Collects per-step (D_gt, D_hat/hidden) trajectories for:
#   - OBS      (lambda=0, no supervision)
#   - OBS_MSE  (lambda=1, supervised)
#   - RAFT     (GRU-64-AC, no explicit prediction)
#
# All 3 modes (DEG / DEAD / STK) × k=1..4 × single episode + aggregate.
# Uses archived paper checkpoints (seed=42 only, representative seed).
#
# Run inside Docker container:
#   bash /root/ws/scripts/experiments/e11_failure_tracking.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"
CKPT_DIR="${REPO_ROOT}/docs/paper_checkpoints"
OUT_BASE="${REPO_ROOT}/docs/results/e11_failure_tracking"
LOG_FILE="${REPO_ROOT}/docs/results/e11_failure_tracking.log"
SEED=42
NUM_ENVS=128

mkdir -p "$OUT_BASE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo " E11 — Failure Tracking Evaluation   $(date)"
echo " Seed: ${SEED}  Envs: ${NUM_ENVS}"
echo "================================================================"

run_tracking() {
    local method="$1" agent="$2" ckpt="$3"
    local out_dir="${OUT_BASE}/${method}/seed_${SEED}"
    mkdir -p "$out_dir"
    echo ""
    echo "── ${method} ──────────────────────────────────────────────"
    echo "  ckpt: ${ckpt}"
    echo "  out:  ${out_dir}"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/eval_failure_tracking.py \
        --task="${OBS_TASK}" \
        --agent="${agent}" \
        --checkpoint="${ckpt}" \
        --method="${method}" \
        --num_envs=${NUM_ENVS} \
        --seed=${SEED} \
        --output_dir="${out_dir}" \
        --headless
}

# ── OBS (lambda=0, no supervised loss) ───────────────────────────────────────
run_tracking "OBS" \
    "rsl_rl_cfg_entry_point" \
    "${CKPT_DIR}/OBS/seed_${SEED}.pt"

# ── OBS-MSE (lambda=1, supervised observer) ───────────────────────────────────
run_tracking "OBS_MSE" \
    "rsl_rl_cfg_entry_point" \
    "${CKPT_DIR}/OBS_MSE/seed_${SEED}.pt"

# ── RAFT (GRU-64-AC, records GRU hidden state) ────────────────────────────────
run_tracking "RAFT" \
    "rsl_rl_rnn_gru64_ac_cfg_entry_point" \
    "${CKPT_DIR}/RAFT/seed_${SEED}.pt"

echo ""
echo "================================================================"
echo " All done.  $(date)"
echo " Results → ${OUT_BASE}"
echo "================================================================"
