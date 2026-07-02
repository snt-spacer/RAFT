#!/usr/bin/env bash
# ============================================================
# E3 — RAFT (GRU-64-AC) failure severity sweep
# Adds RAFT results to docs/results/e3_severity_sweep/raft/{deg,stk}/
# Prerequisites: VAN_GRU64_AC_GoToPosition checkpoints (from E10/E11)
# Run with: bash scripts/experiments/e3_raft_eval.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e3_severity_sweep"
LOG_FILE="${RESULTS_DIR}/raft_eval.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo " E3 — RAFT Failure Severity Sweep        $(date)"
echo " Policy: RAFT (GRU-64-AC, VAN_GRU64_AC_GoToPosition)"
echo " E3a  DEG  scale  = 1.0 0.9 0.7 0.5 0.3 0.1 0.0  (k=1)"
echo " E3b  STK  offset = 0.0 0.1 0.2 0.4 0.6 0.8 1.0  (k=1)"
echo "================================================================"

TASK="Isaaclab-RANSv2-Observer-Position-v0"
RAFT_AGENT="rsl_rl_rnn_gru64_ac_cfg_entry_point"
RAFT_EXP="VAN_GRU64_AC_GoToPosition"

DEG_SEVERITIES=(1.0 0.9 0.7 0.5 0.3 0.1 0.0)
STK_SEVERITIES=(0.0 0.1 0.2 0.4 0.6 0.8 1.0)

RAFT_CKPTS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "${RAFT_EXP}" "$SEED" $OBS_CKPT_ITER)
    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
        RAFT_CKPTS+=("$CKPT")
        echo "  Found RAFT checkpoint seed=${SEED}: $CKPT"
    else
        echo "  [WARN] RAFT checkpoint missing seed=${SEED}"
    fi
done

if [ ${#RAFT_CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] No RAFT checkpoints found."
    exit 1
fi

# ── E3a — DEG severity sweep ──────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " E3a  Continuous Degradation (DEG)  k=1"
echo "────────────────────────────────────────"

for i in "${!RAFT_CKPTS[@]}"; do
    CKPT="${RAFT_CKPTS[$i]}"
    SEED="${SEEDS[$i]}"
    OUT_DIR="${RESULTS_DIR}/raft/deg/seed_${SEED}"
    mkdir -p "$OUT_DIR"
    echo "  seed=${SEED} DEG severity sweep..."
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/experiments/eval_severity.py \
        --task="$TASK" \
        --checkpoint="$CKPT" \
        --mode=deg \
        --severities "${DEG_SEVERITIES[@]}" \
        --output_dir="$OUT_DIR" \
        --num_envs=${NUM_EVAL_ENVS} \
        --eval_episodes_per_env=${EVAL_EPISODES} \
        --pos_tol=${POS_TOL} \
        --success_steps=${SUCCESS_STEPS} \
        --agent="${RAFT_AGENT}" \
        --headless
done

# ── E3b — STK severity sweep ──────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " E3b  Stuck-On (STK)  k=1"
echo "────────────────────────────────────────"

for i in "${!RAFT_CKPTS[@]}"; do
    CKPT="${RAFT_CKPTS[$i]}"
    SEED="${SEEDS[$i]}"
    OUT_DIR="${RESULTS_DIR}/raft/stk/seed_${SEED}"
    mkdir -p "$OUT_DIR"
    echo "  seed=${SEED} STK severity sweep..."
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/experiments/eval_severity.py \
        --task="$TASK" \
        --checkpoint="$CKPT" \
        --mode=stk \
        --severities "${STK_SEVERITIES[@]}" \
        --output_dir="$OUT_DIR" \
        --num_envs=${NUM_EVAL_ENVS} \
        --eval_episodes_per_env=${EVAL_EPISODES} \
        --pos_tol=${POS_TOL} \
        --success_steps=${SUCCESS_STEPS} \
        --agent="${RAFT_AGENT}" \
        --headless
done

echo ""
echo "Done. Results: ${RESULTS_DIR}/raft/"
