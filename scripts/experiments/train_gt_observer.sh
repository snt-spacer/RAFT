#!/usr/bin/env bash
# ============================================================
# Train GT-Observer (mixed-mode fault oracle) — 3 seeds, 5000 iters.
#
# Same env + 3-mode failure curriculum as RAFT, but actor sees D_gt (16-dim).
# This is the proper per-mode oracle upper bound for E1, E3, E4.
#
# Run inside Docker:
#   bash /root/ws/scripts/experiments/train_gt_observer.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

TASK="Isaaclab-RANSv2-Observer-Position-v0"
AGENT="rsl_rl_gt_observer_cfg_entry_point"

RESULTS_DIR="${REPO_ROOT}/docs/results/gt_observer_train"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo " GT-Observer training (mixed-mode oracle)  $(date)"
echo " Task   : ${TASK}"
echo " Agent  : ${AGENT}"
echo " Seeds  : ${SEEDS[*]}"
echo " Iters  : 5000 (matches RAFT)"
echo "================================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "── seed=${SEED} ──────────────────────────────────────────────"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/train.py \
        --task="${TASK}" \
        --agent="${AGENT}" \
        --num_envs=${NUM_TRAIN_ENVS} \
        --headless \
        --seed="${SEED}"
done

echo ""
echo "================================================================"
echo " Done.  $(date)"
echo "================================================================"
