#!/usr/bin/env bash
# ============================================================
# E2 — RAFT per-mode generalization evaluation
#
# Adds RAFT as a 5th row in the E2 generalization matrix.
# Evaluates the RAFT (GRU-64-AC) checkpoint on each failure
# mode in isolation at k=2, using the same protocol as the
# original E2 OBS runs.
#
# Output: docs/results/e2_per_mode_isolation/RAFT/eval_{DEG,DEAD,STK}/seed_{N}/
#
# Run with: bash scripts/experiments/e2_raft_eval.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e2_per_mode_isolation"
LOG_FILE="${RESULTS_DIR}/raft_eval.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

TASK="Isaaclab-RANSv2-Observer-Position-v0"
RAFT_AGENT="rsl_rl_rnn_gru64_ac_cfg_entry_point"
RAFT_EXP="VAN_GRU64_AC_GoToPosition"

echo "================================================================"
echo " E2 — RAFT per-mode generalization       $(date)"
echo " Checkpoint : ${RAFT_EXP}  (model_${OBS_CKPT_ITER}.pt)"
echo " Eval modes : DEG-only  DEAD-only  STK-only  (k=2)"
echo " Seeds      : ${SEEDS[*]}"
echo "================================================================"

echo ""
echo "══ Trained: RAFT (GRU-64-AC, mixed modes) ══"

for EVAL_MODE in DEG DEAD STK; do
    echo ""
    echo "  ── Eval mode: ${EVAL_MODE} ──"
    SEED_JSONS=()

    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "$RAFT_EXP" "$SEED" "$OBS_CKPT_ITER")
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "    [WARN] RAFT checkpoint not found: seed=${SEED}"
            continue
        fi

        OUT_DIR="${RESULTS_DIR}/RAFT/eval_${EVAL_MODE}/seed_${SEED}"
        run_eval "$TASK" "$CKPT" "$OUT_DIR" 2 \
            "--agent=${RAFT_AGENT}" \
            "env.task_name=GoToPositionObserver${EVAL_MODE}"
        SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done

    aggregate_jsons "RAFT  Eval=${EVAL_MODE}  k=0..2" "${SEED_JSONS[@]}"
done

echo ""
echo "================================================================"
echo " Done. Results: ${RESULTS_DIR}/RAFT/"
echo "================================================================"
