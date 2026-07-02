#!/usr/bin/env bash
# ============================================================
# E11 — Asymmetric Critic Ablation (fair comparison vs OBS)
#
# Question: Does adding an asymmetric critic (critic sees ground-truth
# degradation state D_gt during training) close the gap between VAN-MLP /
# GRU / LSTM baselines and OBS?
#
# This experiment isolates the contribution of the asymmetric critic
# independently from the Observer head and MSE supervision.
#
# All variants train on Isaaclab-RANSv2-Observer-Position-v0:
#   - Same failure curriculum as OBS (CuboThrusterFailureTraining)
#   - obs["privileged"] (16-dim D_gt) available for the critic
#   - Actor sees only obs["policy"] — no Observer head, no D_hat
#
# Variants:
#   VAN-AC       : MLP actor, asymmetric critic
#   VAN-GRU-64-AC: GRU-64 actor, asymmetric critic
#   VAN-GRU-256-AC: GRU-256 actor, asymmetric critic
#   VAN-LSTM-64-AC: LSTM-64 actor, asymmetric critic
#   VAN-LSTM-256-AC: LSTM-256 actor, asymmetric critic
#
# Run with: bash scripts/experiments/e11_ac_ablation.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e11_ac_ablation"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"

echo "================================================================"
echo " E11 — Asymmetric Critic Ablation         $(date)"
echo ""
echo " Variants: VAN-AC  GRU-64-AC  GRU-256-AC  LSTM-64-AC  LSTM-256-AC"
echo " Seeds: ${SEEDS[*]}"
echo " Iters: ${OBS_CKPT_ITER} (5000-1=4999 checkpoint)"
echo " Env: ${OBS_TASK}"
echo "================================================================"

declare -A V_AGENT V_EXP
V_AGENT[VAN_AC]="rsl_rl_van_ac_cfg_entry_point"
V_AGENT[GRU64_AC]="rsl_rl_rnn_gru64_ac_cfg_entry_point"
V_AGENT[GRU256_AC]="rsl_rl_rnn_gru256_ac_cfg_entry_point"
V_AGENT[LSTM64_AC]="rsl_rl_rnn_lstm64_ac_cfg_entry_point"
V_AGENT[LSTM256_AC]="rsl_rl_rnn_lstm256_ac_cfg_entry_point"

V_EXP[VAN_AC]="VAN_AC_GoToPosition"
V_EXP[GRU64_AC]="VAN_GRU64_AC_GoToPosition"
V_EXP[GRU256_AC]="VAN_GRU256_AC_GoToPosition"
V_EXP[LSTM64_AC]="VAN_LSTM64_AC_GoToPosition"
V_EXP[LSTM256_AC]="VAN_LSTM256_AC_GoToPosition"

# ── Training ──────────────────────────────────────────────────────────────────
for VARIANT in VAN_AC GRU64_AC GRU256_AC LSTM64_AC LSTM256_AC; do
    echo ""
    echo "================================================================"
    echo " TRAIN  ${VARIANT}"
    echo "================================================================"
    for SEED in "${SEEDS[@]}"; do
        if ! run_train "${VARIANT}" "$OBS_TASK" "${V_AGENT[$VARIANT]}" "$SEED" \
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

for VARIANT in VAN_AC GRU64_AC GRU256_AC LSTM64_AC LSTM256_AC; do
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
        # Pass --agent explicitly so eval loads the right model architecture
        run_eval "$OBS_TASK" "$CKPT" "$OUT_DIR" 4 --agent "${V_AGENT[$VARIANT]}"
        SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    if [ ${#SEED_JSONS[@]} -gt 0 ]; then
        aggregate_jsons "${VARIANT}  k=0..4  mixed-modes" "${SEED_JSONS[@]}"
    fi
done

# Reference: OBS from E1
echo ""
echo "──────────────────────  OBS (E1 reference)  ──────────────────────"
OBS_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${REPO_ROOT}/docs/results/e1_main_comparison/OBS/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && OBS_JSONS+=("$F")
done
[ ${#OBS_JSONS[@]} -gt 0 ] && aggregate_jsons "OBS (E1, reference)" "${OBS_JSONS[@]}"

# Reference: VAN-MLP from E1
echo ""
echo "──────────────────────  VAN-MLP (E1 reference)  ──────────────────────"
VAN_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${REPO_ROOT}/docs/results/e1_main_comparison/VAN/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && VAN_JSONS+=("$F")
done
[ ${#VAN_JSONS[@]} -gt 0 ] && aggregate_jsons "VAN-MLP (E1, reference)" "${VAN_JSONS[@]}"

echo ""
echo "Done. Results: ${RESULTS_DIR}"
