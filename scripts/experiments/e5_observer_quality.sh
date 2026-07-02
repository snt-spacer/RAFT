#!/usr/bin/env bash
# ============================================================
# E5 — Observer System-Identification Quality (eval only)
# Prerequisite: E1 (OBS-ALL) and E2 (OBS-DEG/DEAD/STK) must be complete.
# Run with: bash scripts/experiments/e5_observer_quality.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e5_observer_quality"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E5 — Observer System-Identification Quality  $(date)"
echo " Policies : OBS-ALL  OBS-DEG  OBS-DEAD  OBS-STK"
echo " Eval     : k=2, each mode separately"
echo ""
echo " NOTE: Observer MSE (OMSE) is in WandB:"
echo "   project=Observer_GoToPosition  entity=spacer-rl"
echo "   key: Train/observer_mse"
echo "================================================================"

TASK="Isaaclab-RANSv2-Observer-Position-v0"

declare -A POL_EXP
POL_EXP[OBS_ALL]="Observer_GoToPosition"
POL_EXP[OBS_DEG]="Observer_DEGOnly_GoToPosition"
POL_EXP[OBS_DEAD]="Observer_DEADOnly_GoToPosition"
POL_EXP[OBS_STK]="Observer_STKOnly_GoToPosition"

for POLICY in OBS_ALL OBS_DEG OBS_DEAD OBS_STK; do
    echo ""
    echo "══════════════════  ${POLICY}  ══════════════════"

    for EVAL_MODE in DEG DEAD STK; do
        echo ""
        echo "  ── Eval mode: ${EVAL_MODE}  k=2 ──"
        SEED_JSONS=()

        for SEED in "${SEEDS[@]}"; do
            CKPT=$(find_checkpoint "${POL_EXP[$POLICY]}" "$SEED" $OBS_CKPT_ITER)
            if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
                echo "    [WARN] Checkpoint not found: ${POLICY} seed=${SEED}"
                continue
            fi
            OUT_DIR="${RESULTS_DIR}/${POLICY}/eval_${EVAL_MODE}/seed_${SEED}"
            run_eval "$TASK" "$CKPT" "$OUT_DIR" 2 \
                "env.task_name=GoToPositionObserver${EVAL_MODE}"
            SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
        done

        aggregate_jsons "${POLICY}  eval=${EVAL_MODE}  k=2" "${SEED_JSONS[@]}"
    done
done

echo ""
echo "════════════════════════════════════════════════════"
echo " OMSE COLLECTION INSTRUCTIONS"
echo "════════════════════════════════════════════════════"
echo " 1. Open WandB: project=Observer_GoToPosition  entity=spacer-rl"
echo " 2. Filter runs by group:"
echo "    - OBS-ALL  : observer-sysid"
echo "    - OBS-DEG / OBS-DEAD / OBS-STK : observer-mode-isolation"
echo " 3. Export 'Train/observer_mse' curve → record final value in"
echo "    docs/experiments.md  Results Table E5"
echo ""
echo " WandB run names for this session:"
for POLICY in OBS_ALL OBS_DEG OBS_DEAD OBS_STK; do
    for SEED in "${SEEDS[@]}"; do
        RUN_DIR=$(ls -td "${REPO_ROOT}/logs/rsl_rl/${POL_EXP[$POLICY]}"/*_seed_${SEED} 2>/dev/null | head -1)
        if [ -n "$RUN_DIR" ]; then
            echo "   ${POLICY} seed=${SEED} : $(basename "$RUN_DIR")"
        fi
    done
done
echo "════════════════════════════════════════════════════"

echo ""
echo "Done. Results: ${RESULTS_FILE}"
