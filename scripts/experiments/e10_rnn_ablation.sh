#!/usr/bin/env bash
# ============================================================
# E10 — Recurrent Policy Ablation (GRU / LSTM vs VAN-MLP / OBS)
#
# Question: Can a recurrent policy (GRU or LSTM) trained on the
# same failure curriculum as OBS achieve similar adaptation via
# implicit hidden-state accumulation, without an explicit history
# buffer or MSE-supervised observer head?
#
# Variants trained on Isaaclab-RANSv2-Vanilla-Position-v0:
#   VAN-GRU-64   : GRU, hidden_dim=64  (light)
#   VAN-GRU-256  : GRU, hidden_dim=256 (matched to OBS capacity)
#   VAN-LSTM-64  : LSTM, hidden_dim=64
#   VAN-LSTM-256 : LSTM, hidden_dim=256
#
# All trained 3 seeds × 3000 iters, evaluated k=0..4 mixed modes.
#
# Run with: bash scripts/experiments/e10_rnn_ablation.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e10_rnn_ablation"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

VAN_TASK="Isaaclab-RANSv2-Vanilla-Position-v0"

echo "================================================================"
echo " E10 — Recurrent Policy Ablation          $(date)"
echo ""
echo " Variants: VAN-GRU-64  VAN-GRU-256  VAN-LSTM-64  VAN-LSTM-256"
echo " Seeds: ${SEEDS[*]}"
echo " Iters: ${OBS_CKPT_ITER} (5000-1=4999 checkpoint)"
echo "================================================================"

declare -A V_AGENT V_EXP
V_AGENT[GRU64]="rsl_rl_rnn_gru64_cfg_entry_point"
V_AGENT[GRU256]="rsl_rl_rnn_gru256_cfg_entry_point"
V_AGENT[LSTM64]="rsl_rl_rnn_lstm64_cfg_entry_point"
V_AGENT[LSTM256]="rsl_rl_rnn_lstm256_cfg_entry_point"

V_EXP[GRU64]="VAN_GRU64_GoToPosition"
V_EXP[GRU256]="VAN_GRU256_GoToPosition"
V_EXP[LSTM64]="VAN_LSTM64_GoToPosition"
V_EXP[LSTM256]="VAN_LSTM256_GoToPosition"

# ── Training ──────────────────────────────────────────────────────────────────
for VARIANT in GRU64 GRU256 LSTM64 LSTM256; do
    echo ""
    echo "================================================================"
    echo " TRAIN  ${VARIANT}"
    echo "================================================================"
    for SEED in "${SEEDS[@]}"; do
        if ! run_train "${VARIANT}" "$VAN_TASK" "${V_AGENT[$VARIANT]}" "$SEED" \
            "agent.experiment_name=${V_EXP[$VARIANT]}"; then
            echo "[WARN] Training failed: ${VARIANT} seed=${SEED} — continuing"
        fi
    done
done

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " EVALUATION  (k=0..4, mixed modes)"
echo "================================================================"

for VARIANT in GRU64 GRU256 LSTM64 LSTM256; do
    echo ""
    echo "──────────────────────  ${VARIANT}  ──────────────────────"
    SEED_JSONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${V_EXP[$VARIANT]}" "$SEED" "$OBS_CKPT_ITER")
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [WARN] Checkpoint not found: ${VARIANT} seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/${VARIANT}/seed_${SEED}"
        run_eval "$VAN_TASK" "$CKPT" "$OUT_DIR" 4
        SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    if [ ${#SEED_JSONS[@]} -gt 0 ]; then
        aggregate_jsons "${VARIANT}  k=0..4  mixed-modes" "${SEED_JSONS[@]}"
    fi
done

# Reference: VAN-MLP and OBS from E1
echo ""
echo "──────────────────────  VAN-MLP (E1 reference)  ──────────────────────"
VAN_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${REPO_ROOT}/docs/results/e1_main_comparison/VAN/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && VAN_JSONS+=("$F")
done
[ ${#VAN_JSONS[@]} -gt 0 ] && aggregate_jsons "VAN-MLP (E1, reference)" "${VAN_JSONS[@]}"

echo ""
echo "──────────────────────  OBS (E1 reference)  ──────────────────────"
OBS_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${REPO_ROOT}/docs/results/e1_main_comparison/OBS/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && OBS_JSONS+=("$F")
done
[ ${#OBS_JSONS[@]} -gt 0 ] && aggregate_jsons "OBS (E1, reference)" "${OBS_JSONS[@]}"

echo ""
echo "Done. Results: ${RESULTS_DIR}"
