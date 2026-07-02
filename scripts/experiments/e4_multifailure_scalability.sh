#!/usr/bin/env bash
# ============================================================
# E3 — Simultaneous Multi-Thruster Failure Scalability (eval only)
# Prerequisites:
#   - OBS (λ=0, ABL_NoMSE) checkpoints in logs/rsl_rl/Observer_ABL_NoMSE/
#   - GT-Observer checkpoints in logs/rsl_rl/GT_OBS_LongTrain/ (mixed-mode oracle)
#
# All three policies (RAFT, OBS, GT-Obs) are evaluated per-mode with mode pinning.
#
# Run with: bash scripts/experiments/e4_multifailure_scalability.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e4_multifailure_scalability"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E3 — Multi-Thruster Failure Scalability $(date)"
echo " Policies : OBS (λ=0)  GT-Observer (mixed-mode oracle)"
echo " k sweep  : 0..4"
echo " Modes    : DEG  DEAD  STK  (each evaluated separately)"
echo "================================================================"

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"
GT_OBS_AGENT="rsl_rl_gt_observer_cfg_entry_point"

for EVAL_MODE in DEG DEAD STK; do
    echo ""
    echo "────────────────────────────────────────────────"
    echo " Mode: ${EVAL_MODE}"
    echo "────────────────────────────────────────────────"

    # ── OBS (λ=0, ABL_NoMSE) ──────────────────────────────────────────────
    echo ""
    echo "  Policy: OBS (λ=0)"
    OBS_JSONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "Observer_ABL_NoMSE" "$SEED" $OBS_CKPT_ITER)
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "    [WARN] OBS-ALL checkpoint missing seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/OBS/${EVAL_MODE}/seed_${SEED}"
        run_eval "$OBS_TASK" "$CKPT" "$OUT_DIR" 4 \
            "env.task_name=GoToPositionObserver${EVAL_MODE}"
        OBS_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    aggregate_jsons "OBS (λ=0)  mode=${EVAL_MODE}  k=0..4" "${OBS_JSONS[@]}"

    # ── GT-Observer (mixed-mode oracle, actor sees D_gt) ───────────────────
    # Trained on Observer-Position-v0 with the same 3-mode failure curriculum
    # as RAFT/OBS, but the actor receives the 16-dim D_gt. Evaluated per-mode
    # via env.task_name pinning, exactly like RAFT and OBS.
    echo ""
    echo "  Policy: GT-Observer  mode=${EVAL_MODE}"
    GT_OBS_JSONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "GT_OBS_LongTrain" "$SEED" $OBS_CKPT_ITER)
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "    [WARN] GT-Observer checkpoint missing seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/GT/${EVAL_MODE}/seed_${SEED}"
        run_eval "$OBS_TASK" "$CKPT" "$OUT_DIR" 4 \
            "--agent=${GT_OBS_AGENT}" \
            "env.task_name=GoToPositionObserver${EVAL_MODE}"
        GT_OBS_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    aggregate_jsons "GT-Observer  mode=${EVAL_MODE}  k=0..4" "${GT_OBS_JSONS[@]}"
done

echo ""
echo "Done. Results: ${RESULTS_FILE}"
