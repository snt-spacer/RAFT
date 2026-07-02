#!/usr/bin/env bash
# ============================================================
# E4 — RAFT (GRU-64-AC) per-mode scalability eval
# Adds RAFT results to docs/results/e4_multifailure_scalability/RAFT/
# Prerequisites: VAN_GRU64_AC_GoToPosition checkpoints (from E10/E11)
# Run with: bash scripts/experiments/e4_raft_eval.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e4_multifailure_scalability"
LOG_FILE="${RESULTS_DIR}/raft_eval.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo " E4 — RAFT per-mode scalability eval     $(date)"
echo " Policy: RAFT (GRU-64-AC, VAN_GRU64_AC_GoToPosition)"
echo " Modes : DEG  DEAD  STK   k=0..4"
echo "================================================================"

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"
RAFT_AGENT="rsl_rl_rnn_gru64_ac_cfg_entry_point"
RAFT_EXP="VAN_GRU64_AC_GoToPosition"

for EVAL_MODE in DEG DEAD STK; do
    echo ""
    echo "────────────────────────────────────────────────"
    echo " Mode: ${EVAL_MODE}"
    echo "────────────────────────────────────────────────"

    RAFT_JSONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${RAFT_EXP}" "$SEED" $OBS_CKPT_ITER)
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "    [WARN] RAFT checkpoint missing seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/RAFT/${EVAL_MODE}/seed_${SEED}"
        run_eval "$OBS_TASK" "$CKPT" "$OUT_DIR" 4 \
            "env.task_name=GoToPositionObserver${EVAL_MODE}" \
            --agent "${RAFT_AGENT}"
        RAFT_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    aggregate_jsons "RAFT (GRU-64-AC)  mode=${EVAL_MODE}  k=0..4" "${RAFT_JSONS[@]}"
done

echo ""
echo "Done. Results: ${RESULTS_DIR}/RAFT/"
