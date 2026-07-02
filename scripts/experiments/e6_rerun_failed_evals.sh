#!/usr/bin/env bash
# Re-runs only the E6 evals that failed due to architecture mismatch:
#   ABL_HIST16, ABL_HIST128, ABL_SMALLOBS
# OBS_FULL and ABL_NOMSE already have valid results and are skipped.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e6_ablation"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
TASK="Isaaclab-RANSv2-Observer-Position-v0"

exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E6 re-run: ABL_HIST16  ABL_HIST128  ABL_SMALLOBS  $(date)"
echo "================================================================"

declare -A ABL_EXP ABL_EVAL_EXTRA
ABL_EXP[ABL_HIST16]="Observer_ABL_Hist16"
ABL_EXP[ABL_HIST128]="Observer_ABL_Hist128"
ABL_EXP[ABL_SMALLOBS]="Observer_ABL_SmallObs"

ABL_EVAL_EXTRA[ABL_HIST16]="env.history_len=16"
ABL_EVAL_EXTRA[ABL_HIST128]="env.history_len=128"
ABL_EVAL_EXTRA[ABL_SMALLOBS]="agent.actor.observer_hidden_dims=[32,16]"

for VARIANT in ABL_HIST16 ABL_HIST128 ABL_SMALLOBS; do
    echo ""
    echo "──────────────────────  ${VARIANT}  ──────────────────────"
    SEED_JSONS=()

    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${ABL_EXP[$VARIANT]}" "$SEED" $OBS_CKPT_ITER)
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "    [WARN] Checkpoint not found: ${VARIANT} seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/${VARIANT}/seed_${SEED}"
        run_eval "$TASK" "$CKPT" "$OUT_DIR" 4 ${ABL_EVAL_EXTRA[$VARIANT]}
        SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done

    aggregate_jsons "${VARIANT}  k=0..4  mixed-modes" "${SEED_JSONS[@]}"
done

echo ""
echo "Done. Full results: ${RESULTS_FILE}"
