#!/usr/bin/env bash
# ============================================================
# E2 — Per-Mode Isolation Study
# Prerequisites:
#   - E6 ABL_NOMSE (OBS, λ=0) in logs/rsl_rl/Observer_ABL_NoMSE/
#   - Mode specialists in logs/rsl_rl/Observer_{DEG,DEAD,STK}Only_GoToPosition/
# Run with: bash scripts/experiments/e2_per_mode_isolation.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e2_per_mode_isolation"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E2 — Per-Mode Isolation                 $(date)"
echo " Train policies : OBS-ALL  OBS-DEG  OBS-DEAD  OBS-STK"
echo " Eval modes     : DEG-only  DEAD-only  STK-only  (k=2)"
echo "================================================================"

TASK="Isaaclab-RANSv2-Observer-Position-v0"

declare -A M_EXP M_TRAIN_EXTRA
M_EXP[OBS_ALL]="Observer_ABL_NoMSE"             # OBS (λ=0, proposed) — skip training
M_EXP[OBS_DEG]="Observer_DEGOnly_GoToPosition"
M_EXP[OBS_DEAD]="Observer_DEADOnly_GoToPosition"
M_EXP[OBS_STK]="Observer_STKOnly_GoToPosition"

M_TRAIN_EXTRA[OBS_DEG]="agent.experiment_name=Observer_DEGOnly_GoToPosition env.task_name=GoToPositionObserverDEG"
M_TRAIN_EXTRA[OBS_DEAD]="agent.experiment_name=Observer_DEADOnly_GoToPosition env.task_name=GoToPositionObserverDEAD"
M_TRAIN_EXTRA[OBS_STK]="agent.experiment_name=Observer_STKOnly_GoToPosition env.task_name=GoToPositionObserverSTK"

# ── Training (skip if checkpoint already exists) ──────────────────────────────
for METHOD in OBS_DEG OBS_DEAD OBS_STK; do
    for SEED in "${SEEDS[@]}"; do
        EXISTING=$(find_checkpoint "${M_EXP[$METHOD]}" "$SEED" $OBS_CKPT_ITER)
        if [ -n "$EXISTING" ] && [ -f "$EXISTING" ]; then
            echo "  [SKIP] ${METHOD} seed=${SEED} — checkpoint exists"
            continue
        fi
        if ! run_train "$METHOD" "$TASK" "rsl_rl_cfg_entry_point" "$SEED" \
            ${M_TRAIN_EXTRA[$METHOD]}; then
            echo "[WARN] Training failed: ${METHOD} seed=${SEED} — continuing"
        fi
    done
done

# ── Evaluation: 4 trained policies × 3 eval modes ────────────────────────────
echo ""
echo "================================================================"
echo " EVALUATION RESULTS  (k=2)"
echo "================================================================"

for TRAIN_METHOD in OBS_ALL OBS_DEG OBS_DEAD OBS_STK; do
    echo ""
    echo "══ Trained: ${TRAIN_METHOD} ══"

    for EVAL_MODE in DEG DEAD STK; do
        echo "  ── Eval mode: ${EVAL_MODE} ──"
        SEED_JSONS=()

        for SEED in "${SEEDS[@]}"; do
            CKPT=$(find_checkpoint "${M_EXP[$TRAIN_METHOD]}" "$SEED" $OBS_CKPT_ITER)
            if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
                echo "    [WARN] Checkpoint not found: ${TRAIN_METHOD} seed=${SEED}"
                continue
            fi

            OUT_DIR="${RESULTS_DIR}/${TRAIN_METHOD}/eval_${EVAL_MODE}/seed_${SEED}"
            run_eval "$TASK" "$CKPT" "$OUT_DIR" 2 \
                "env.task_name=GoToPositionObserver${EVAL_MODE}"
            SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
        done

        aggregate_jsons "Train=${TRAIN_METHOD}  Eval=${EVAL_MODE}  k=2" "${SEED_JSONS[@]}"
    done
done

echo ""
echo "Done. Results: ${RESULTS_FILE}"
