#!/usr/bin/env bash
# ============================================================
# E4 re-eval: new GT-Observer oracle, mid-episode failure injection.
#
# Evaluates the new GT-Observer checkpoints (logs/rsl_rl/GT_OBS_LongTrain/)
# under mid-episode injection at step=100, mixed-mode failures, k=1..4.
# Output: docs/results/e9_mid_episode/GT_OBS/seed_*/
#
# Run inside Docker:
#   bash /root/ws/scripts/experiments/e4_gt_observer_mid_episode.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"
GT_OBS_AGENT="rsl_rl_gt_observer_cfg_entry_point"
INJECT_STEP=100

RESULTS_DIR="${REPO_ROOT}/docs/results/e9_mid_episode"
RESULTS_FILE="${RESULTS_DIR}/gt_observer_eval.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E4 re-eval: GT-Observer mid-episode injection  $(date)"
echo " inject_step : ${INJECT_STEP}"
echo "================================================================"

run_mid_eval() {
    local task="$1" agent="$2" ckpt="$3" out_dir="$4" max_k="$5"
    mkdir -p "$out_dir"
    echo "  mid-eval  task=${task}  agent=${agent}  k=1..${max_k}  inject=${INJECT_STEP}  ckpt=$(basename "$ckpt")"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/eval_mid_episode_failures.py \
        --task="${task}" \
        --agent="${agent}" \
        --checkpoint="${ckpt}" \
        --num_envs=${NUM_EVAL_ENVS} \
        --eval_episodes_per_env=${EVAL_EPISODES} \
        --max_failures="${max_k}" \
        --inject_step=${INJECT_STEP} \
        --pos_tol=${POS_TOL} \
        --success_steps=${SUCCESS_STEPS} \
        --output_dir="${out_dir}" \
        --headless
}

GT_OBS_JSONS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "GT_OBS_LongTrain" "$SEED" $OBS_CKPT_ITER)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "  [WARN] GT-Observer checkpoint missing seed=${SEED}"
        continue
    fi
    OUT_DIR="${RESULTS_DIR}/GT_OBS/seed_${SEED}"
    run_mid_eval "$OBS_TASK" "$GT_OBS_AGENT" "$CKPT" "$OUT_DIR" 4
    GT_OBS_JSONS+=("${OUT_DIR}/eval_mid_episode.json")
done

echo ""
echo "Done. Results: ${RESULTS_DIR}"
