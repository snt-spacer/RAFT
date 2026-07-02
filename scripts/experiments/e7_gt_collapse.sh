#!/usr/bin/env bash
# ============================================================
# E7 — GT Collapse Investigation
#
# Root cause: the standard GT run (2000 iters × 16 steps × 4096 envs ≈ 131M steps)
# never advances its curriculum past k=2, because warmup+ramp = 250M steps.
#
#   k_max at iter 2000 = round((131M − 50M) / 200M × 4) = 2
#
# Two variants to isolate the cause:
#
#   GT-LONGTR:  same GT config, max_iterations=5000 (≈327M steps, curriculum
#               completes at ~iter 3814, leaving 1186 iters with k_max=4).
#
#   GT-NOCURR:  new robot config "CuboThrusterFailureNoCurr" (warmup=0, ramp=1),
#               k=4 from step 1, 2000 iters (same budget as original GT).
#
# Both are trained 3 seeds and evaluated k=0..4.
# Run with: bash scripts/experiments/e7_gt_collapse.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e7_gt_collapse"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

GT_TASK="Isaaclab-RANSv2-GroundTruth-Position-v0"

echo "================================================================"
echo " E7 — GT Collapse Investigation          $(date)"
echo ""
echo " GT budget analysis:"
echo "   Standard GT: 2000 iters × 16 steps × 4096 envs = 131M steps"
echo "   Curriculum:  warmup=50M + ramp=200M = 250M to complete"
echo "   k_max at end of standard GT ≈ round((131M-50M)/200M × 4) = 2"
echo "   → GT evaluated at k=3,4 but NEVER trained with k>2"
echo ""
echo " Variants:"
echo "   GT-LONGTR : 5000 iters (≈327M steps, curriculum completes at iter ~3814)"
echo "   GT-NOCURR : 2000 iters, no curriculum (k=4 from step 1)"
echo "================================================================"

# ── GT-LONGTR ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " GT-LONGTR — extended training (5000 iters, standard curriculum)"
echo "================================================================"
for SEED in "${SEEDS[@]}"; do
    if ! run_train "GT-LONGTR" "$GT_TASK" "rsl_rl_cfg_entry_point" "$SEED" \
        "env.robot_name=CuboThrusterFailureTraining" \
        "agent.experiment_name=GT_LongTrain" \
        "agent.max_iterations=5000"; then
        echo "[WARN] Training failed: GT-LONGTR seed=${SEED} — continuing"
    fi
done

# ── GT-NOCURR ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " GT-NOCURR — no curriculum (k=4 from step 1, 2000 iters)"
echo "================================================================"
for SEED in "${SEEDS[@]}"; do
    if ! run_train "GT-NOCURR" "$GT_TASK" "rsl_rl_cfg_entry_point" "$SEED" \
        "env.robot_name=CuboThrusterFailureNoCurr" \
        "agent.experiment_name=GT_NoCurr"; then
        echo "[WARN] Training failed: GT-NOCURR seed=${SEED} — continuing"
    fi
done

# ── Evaluation ───────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " EVALUATION  (k=0..4, mixed modes)"
echo "================================================================"

declare -A VAR_EXP VAR_CKPT_ITER
VAR_EXP[GT_LONGTR]="GT_LongTrain"
VAR_EXP[GT_NOCURR]="GT_NoCurr"
VAR_CKPT_ITER[GT_LONGTR]=4999
VAR_CKPT_ITER[GT_NOCURR]=$GT_CKPT_ITER  # 1999

for VARIANT in GT_LONGTR GT_NOCURR; do
    echo ""
    echo "──────────────────────  ${VARIANT}  ──────────────────────"
    SEED_JSONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${VAR_EXP[$VARIANT]}" "$SEED" "${VAR_CKPT_ITER[$VARIANT]}")
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [WARN] Checkpoint not found: ${VARIANT} seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/${VARIANT}/seed_${SEED}"
        run_eval "$GT_TASK" "$CKPT" "$OUT_DIR" 4
        SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    if [ ${#SEED_JSONS[@]} -gt 0 ]; then
        aggregate_jsons "${VARIANT}  k=0..4  mixed-modes" "${SEED_JSONS[@]}"
    fi
done

# Also summarize original GT results for comparison
echo ""
echo "──────────────────────  GT-ORIGINAL (reference, from E1)  ──────────────────────"
ORIG_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${REPO_ROOT}/docs/results/e1_main_comparison/GT/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && ORIG_JSONS+=("$F")
done
if [ ${#ORIG_JSONS[@]} -gt 0 ]; then
    aggregate_jsons "GT-ORIGINAL  (2000 iters, curriculum stops at k=2)" "${ORIG_JSONS[@]}"
fi

echo ""
echo "Done. Results: ${RESULTS_FILE}"
