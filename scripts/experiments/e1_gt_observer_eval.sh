#!/usr/bin/env bash
# ============================================================
# E1 re-eval: new GT-Observer oracle under mixed-mode failures.
#
# Evaluates the new GT-Observer checkpoints (logs/rsl_rl/GT_OBS_LongTrain/)
# under mixed-mode failures (default failure_mode_probs in ObserverEnvCfg).
# Writes results to docs/results/e1_main_comparison/GT_OBS/seed_*/.
#
# Run inside Docker:
#   bash /root/ws/scripts/experiments/e1_gt_observer_eval.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"
GT_OBS_AGENT="rsl_rl_gt_observer_cfg_entry_point"

RESULTS_DIR="${REPO_ROOT}/docs/results/e1_main_comparison"
RESULTS_FILE="${RESULTS_DIR}/gt_observer_eval.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E1 re-eval: GT-Observer (mixed-mode oracle)  $(date)"
echo "================================================================"

GT_OBS_JSONS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "GT_OBS_LongTrain" "$SEED" $OBS_CKPT_ITER)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "  [WARN] GT-Observer checkpoint missing seed=${SEED}"
        continue
    fi
    OUT_DIR="${RESULTS_DIR}/GT_OBS/seed_${SEED}"
    run_eval "$OBS_TASK" "$CKPT" "$OUT_DIR" 4 \
        "--agent=${GT_OBS_AGENT}"
    GT_OBS_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
done

aggregate_jsons "GT-Observer (mixed-mode oracle)  k=0..4" "${GT_OBS_JSONS[@]}"

echo ""
echo "Done. Results: ${RESULTS_FILE}"
