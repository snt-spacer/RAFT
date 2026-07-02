#!/usr/bin/env bash
# ============================================================
# E3 — Failure Severity Sweep (eval only — no training needed)
# Prerequisite: E6 ABL_NOMSE (OBS, λ=0) in logs/rsl_rl/Observer_ABL_NoMSE/
# Run with: bash scripts/experiments/e3_severity_sweep.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e3_severity_sweep"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E3 — Failure Severity Sweep             $(date)"
echo " Policy: OBS (λ=0, ABL_NoMSE)"
echo " E3a  DEG  scale  = 1.0 0.9 0.7 0.5 0.3 0.1 0.0  (k=1)"
echo " E3b  STK  offset = 0.0 0.1 0.2 0.4 0.6 0.8 1.0  (k=1)"
echo "================================================================"

TASK="Isaaclab-RANSv2-Observer-Position-v0"
DEG_SEVERITIES=(1.0 0.9 0.7 0.5 0.3 0.1 0.0)
STK_SEVERITIES=(0.0 0.1 0.2 0.4 0.6 0.8 1.0)

# ── Collect OBS-ALL checkpoints from E1 ───────────────────────────────────
OBS_CKPTS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "Observer_ABL_NoMSE" "$SEED" $OBS_CKPT_ITER)
    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
        OBS_CKPTS+=("$CKPT")
        echo "  Found E1 checkpoint seed=${SEED}: $CKPT"
    else
        echo "  [WARN] E1 checkpoint missing for seed=${SEED}"
    fi
done

if [ ${#OBS_CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] No OBS-ALL checkpoints found. Run E1 first."
    exit 1
fi

# ── E3a — DEG severity sweep ──────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " E3a  Continuous Degradation (DEG)  k=1"
echo "────────────────────────────────────────"

DEG_SEED_DIRS=()
for i in "${!OBS_CKPTS[@]}"; do
    CKPT="${OBS_CKPTS[$i]}"
    SEED="${SEEDS[$i]}"
    OUT_DIR="${RESULTS_DIR}/deg/seed_${SEED}"
    mkdir -p "$OUT_DIR"
    echo "  Evaluating seed=${SEED} DEG severity sweep..."
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
        --headless
    DEG_SEED_DIRS+=("${OUT_DIR}/severity_results.json")
done

aggregate_jsons "E3a DEG severity sweep (k=1)" "${DEG_SEED_DIRS[@]}"

# ── E3b — STK severity sweep ──────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " E3b  Stuck-On (STK)  k=1"
echo "────────────────────────────────────────"

STK_SEED_DIRS=()
for i in "${!OBS_CKPTS[@]}"; do
    CKPT="${OBS_CKPTS[$i]}"
    SEED="${SEEDS[$i]}"
    OUT_DIR="${RESULTS_DIR}/stk/seed_${SEED}"
    mkdir -p "$OUT_DIR"
    echo "  Evaluating seed=${SEED} STK severity sweep..."
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
        --headless
    STK_SEED_DIRS+=("${OUT_DIR}/severity_results.json")
done

aggregate_jsons "E3b STK severity sweep (k=1)" "${STK_SEED_DIRS[@]}"

echo ""
echo "Done. Results: ${RESULTS_FILE}"
