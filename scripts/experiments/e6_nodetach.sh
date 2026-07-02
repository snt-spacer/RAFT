#!/usr/bin/env bash
# ============================================================
# E6 — ABL-NODETACH ablation
#
# Temporarily removes .detach() from ObserverActorModel.get_latent(),
# trains 3 seeds, evaluates, then restores the original file.
# The trap ensures the restore runs even if the script is interrupted.
#
# Run with: bash scripts/experiments/e6_nodetach.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e6_ablation"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

OBSERVER_MODEL="/root/rsl_rl/rsl_rl/models/observer_model.py"
TASK="Isaaclab-RANSv2-Observer-Position-v0"
EXP_NAME="Observer_ABL_NoDetach"

echo "================================================================"
echo " E6 — ABL-NODETACH                       $(date)"
echo " Patching: ${OBSERVER_MODEL}"
echo "================================================================"

# ── Sanity-check the file exists and contains the expected line ───────────────
if [ ! -f "$OBSERVER_MODEL" ]; then
    echo "[ERROR] observer_model.py not found at ${OBSERVER_MODEL}"
    exit 1
fi

if ! grep -q "get_observer_output(obs).detach()" "$OBSERVER_MODEL"; then
    echo "[ERROR] Expected line not found — has the file already been patched?"
    grep -n "detach\|get_observer_output" "$OBSERVER_MODEL"
    exit 1
fi

# ── Patch + restore trap ──────────────────────────────────────────────────────
restore_detach() {
    echo ""
    echo ">>> Restoring .detach() in observer_model.py ..."
    sed -i 's/d_hat = self\.get_observer_output(obs)$/d_hat = self.get_observer_output(obs).detach()/' \
        "$OBSERVER_MODEL"
    if grep -q "get_observer_output(obs).detach()" "$OBSERVER_MODEL"; then
        echo ">>> Restored successfully."
    else
        echo "[WARN] Restore may have failed — check ${OBSERVER_MODEL} manually."
    fi
}
trap restore_detach EXIT

echo ""
echo ">>> Removing .detach() ..."
sed -i 's/d_hat = self\.get_observer_output(obs)\.detach()/d_hat = self.get_observer_output(obs)/' \
    "$OBSERVER_MODEL"
grep -n "get_observer_output" "$OBSERVER_MODEL"   # confirm patch

# ── Training ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " TRAINING  (3 seeds)"
echo "================================================================"
for SEED in "${SEEDS[@]}"; do
    if ! run_train "ABL_NODETACH" "$TASK" "rsl_rl_cfg_entry_point" "$SEED" \
        "agent.experiment_name=${EXP_NAME}"; then
        echo "[WARN] Training failed: ABL_NODETACH seed=${SEED} — continuing"
    fi
done

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " EVALUATION  (k=0..4, mixed modes)"
echo "================================================================"
SEED_JSONS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "$EXP_NAME" "$SEED" $OBS_CKPT_ITER)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "  [WARN] Checkpoint not found: ABL_NODETACH seed=${SEED}"
        continue
    fi
    OUT_DIR="${RESULTS_DIR}/ABL_NODETACH/seed_${SEED}"
    # No arch override needed — same shape as OBS_FULL
    run_eval "$TASK" "$CKPT" "$OUT_DIR" 4
    SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
done

aggregate_jsons "ABL_NODETACH  k=0..4  mixed-modes" "${SEED_JSONS[@]}"

echo ""
echo "Done. Results: ${RESULTS_FILE}"
# trap fires here and restores the file
