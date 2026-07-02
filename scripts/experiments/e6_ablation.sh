#!/usr/bin/env bash
# ============================================================
# E6 — Ablation Study
# Prerequisite: E1 must have been run (OBS-FULL checkpoints reused).
# Run with: bash scripts/experiments/e6_ablation.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e6_ablation"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "================================================================"
echo " E6 — Ablation Study                     $(date)"
echo " Variants : OBS-FULL  ABL-NOMSE  ABL-HIST16  ABL-HIST128  ABL-SMALLOBS"
echo " Eval     : k=4  mixed modes"
echo "================================================================"

TASK="Isaaclab-RANSv2-Observer-Position-v0"

declare -A ABL_EXP ABL_EXTRA
ABL_EXP[OBS_FULL]="Observer_GoToPosition"          # reuse E1 — skip training
ABL_EXP[ABL_NOMSE]="Observer_ABL_NoMSE"
ABL_EXP[ABL_HIST16]="Observer_ABL_Hist16"
ABL_EXP[ABL_HIST128]="Observer_ABL_Hist128"
ABL_EXP[ABL_SMALLOBS]="Observer_ABL_SmallObs"

ABL_EXTRA[ABL_NOMSE]="agent.experiment_name=Observer_ABL_NoMSE agent.algorithm.observer_loss_coef=0.0"
ABL_EXTRA[ABL_HIST16]="agent.experiment_name=Observer_ABL_Hist16 env.history_len=16"
ABL_EXTRA[ABL_HIST128]="agent.experiment_name=Observer_ABL_Hist128 env.history_len=128"
ABL_EXTRA[ABL_SMALLOBS]="agent.experiment_name=Observer_ABL_SmallObs agent.actor.observer_hidden_dims=[32,16]"

# Architecture overrides needed at eval time to reconstruct the correct model shape.
# Training-only flags (experiment_name, observer_loss_coef) are omitted here.
declare -A ABL_EVAL_EXTRA
ABL_EVAL_EXTRA[OBS_FULL]=""
ABL_EVAL_EXTRA[ABL_NOMSE]=""
ABL_EVAL_EXTRA[ABL_HIST16]="env.history_len=16"
ABL_EVAL_EXTRA[ABL_HIST128]="env.history_len=128"
ABL_EVAL_EXTRA[ABL_SMALLOBS]="agent.actor.observer_hidden_dims=[32,16]"

# ── Training (skip OBS-FULL — reuse E1) ──────────────────────────────────────
for VARIANT in ABL_NOMSE ABL_HIST16 ABL_HIST128 ABL_SMALLOBS; do
    for SEED in "${SEEDS[@]}"; do
        if ! run_train "$VARIANT" "$TASK" "rsl_rl_cfg_entry_point" "$SEED" \
            ${ABL_EXTRA[$VARIANT]}; then
            echo "[WARN] Training failed: ${VARIANT} seed=${SEED} — continuing"
        fi
    done
done

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " EVALUATION RESULTS  (k=4, mixed modes)"
echo "================================================================"

for VARIANT in OBS_FULL ABL_NOMSE ABL_HIST16 ABL_HIST128 ABL_SMALLOBS; do
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
echo "════════════════════════════════════════════════════"
echo " ABL-NODETACH — MANUAL STEPS REQUIRED"
echo "════════════════════════════════════════════════════"
echo " 1. Edit rsl_rl/rsl_rl/models/observer_model.py"
echo "    In get_latent(), change:"
echo "      d_hat = self.get_observer_output(obs).detach()"
echo "    to:"
echo "      d_hat = self.get_observer_output(obs)   # no detach"
echo " 2. Train 3 seeds:"
for SEED in "${SEEDS[@]}"; do
echo "      ${PYTHON_EXE} scripts/rsl_rl/train.py --task=${TASK} --num_envs=${NUM_TRAIN_ENVS} --headless --seed=${SEED} agent.experiment_name=Observer_ABL_NoDetach"
done
echo " 3. Eval and aggregate:"
echo "      for SEED in ${SEEDS[*]}; do"
echo "        CKPT=\$(ls -td logs/rsl_rl/Observer_ABL_NoDetach/*_seed_\${SEED}/ | head -1)model_${OBS_CKPT_ITER}.pt"
echo "        ${PYTHON_EXE} scripts/rsl_rl/eval_gt_failures.py --task=${TASK} --checkpoint=\$CKPT --num_envs=${NUM_EVAL_ENVS} --max_failures=4 --eval_episodes_per_env=${EVAL_EPISODES} --headless --output_dir=docs/results/e6_ablation/ABL_NODETACH/seed_\${SEED}"
echo "      done"
echo " 4. RESTORE .detach() in observer_model.py"
echo "════════════════════════════════════════════════════"

echo ""
echo "Done. Results: ${RESULTS_FILE}"
