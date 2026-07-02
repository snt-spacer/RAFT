#!/usr/bin/env bash
# ============================================================
# E1 — Main Comparison: VAN, GT, OBS across k=0..4
# (HIS excluded — Transformer baseline did not converge)
#
# Trains all three methods (3 seeds each) then evaluates every
# checkpoint under mixed-mode failures with k swept 0..4.
#
# Run with:
#   bash scripts/experiments/e1_main_comparison.sh
#
# All output is automatically saved to:
#   docs/results/e1_main_comparison/summary.txt
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e1_main_comparison"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E1 — Main Comparison                    $(date)"
echo " Methods : VAN  GT  OBS  (HIS excluded)"
echo " Seeds   : ${SEEDS[*]}"
echo " k sweep : 0..4  (mixed failure modes)"
echo "================================================================"

# ── Per-method config ────────────────────────────────────────────────────────
declare -A M_TASK M_AGENT M_EXP M_CKPT_ITER M_TRAIN_EXTRA

# VAN — plain MLP, no history encoding, no failures during training.
# Uses Isaaclab-RANSv2-Vanilla-Position-v0 which points to the vanilla MLP cfg.
M_TASK[VAN]="Isaaclab-RANSv2-Vanilla-Position-v0"
M_AGENT[VAN]="rsl_rl_cfg_entry_point"
M_EXP[VAN]="Vanilla_GoToPosition"
M_CKPT_ITER[VAN]=$HIS_CKPT_ITER
M_TRAIN_EXTRA[VAN]="agent.experiment_name=Vanilla_GoToPosition"

# GT — oracle upper bound (sees D_gt directly in obs)
M_TASK[GT]="Isaaclab-RANSv2-GroundTruth-Position-v0"
M_AGENT[GT]="rsl_rl_cfg_entry_point"
M_EXP[GT]="AutoEnvGen_RNN_GroundTruthObs"
M_CKPT_ITER[GT]=$GT_CKPT_ITER
M_TRAIN_EXTRA[GT]="env.robot_name=CuboThrusterFailureTraining"

# OBS — proposed method (ObserverEnvCfg already sets CuboThrusterFailureTraining)
M_TASK[OBS]="Isaaclab-RANSv2-Observer-Position-v0"
M_AGENT[OBS]="rsl_rl_cfg_entry_point"
M_EXP[OBS]="Observer_GoToPosition"
M_CKPT_ITER[OBS]=$OBS_CKPT_ITER
M_TRAIN_EXTRA[OBS]=""

# ── Training ─────────────────────────────────────────────────────────────────
for METHOD in VAN GT OBS; do
    for SEED in "${SEEDS[@]}"; do
        if ! run_train "$METHOD" \
            "${M_TASK[$METHOD]}" "${M_AGENT[$METHOD]}" "$SEED" \
            ${M_TRAIN_EXTRA[$METHOD]}; then
            echo "[WARN] Training failed: ${METHOD} seed=${SEED} — continuing"
        fi
    done
done

# ── Evaluation ───────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " EVALUATION RESULTS"
echo "================================================================"

declare -A METHOD_JSON_FILES
for METHOD in VAN GT OBS; do
    echo ""
    echo "──────────────────────  ${METHOD}  ──────────────────────"
    METHOD_JSON_FILES[$METHOD]=""
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${M_EXP[$METHOD]}" "$SEED" "${M_CKPT_ITER[$METHOD]}")
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [WARN] Checkpoint not found for ${METHOD} seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/${METHOD}/seed_${SEED}"
        run_eval "${M_TASK[$METHOD]}" "$CKPT" "$OUT_DIR" 4
        METHOD_JSON_FILES[$METHOD]+=" ${OUT_DIR}/eval_gt_failures.json"
    done
done

# ── Results summary ───────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " AGGREGATED RESULTS  (mean ± std across ${#SEEDS[@]} seeds)"
echo "================================================================"

for METHOD in VAN GT OBS; do
    FILES=(${METHOD_JSON_FILES[$METHOD]})
    if [ ${#FILES[@]} -gt 0 ]; then
        aggregate_jsons "$METHOD  k=0..4  mixed-modes" "${FILES[@]}"
    fi
done

echo ""
echo "Done. Results: ${RESULTS_FILE}"
