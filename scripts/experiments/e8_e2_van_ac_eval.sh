#!/usr/bin/env bash
# ============================================================
# VAN-MLP-AC supplementary evaluations:
#   - E8: mid-episode failure injection (inject_step=100)
#   - E2: per-mode generalization (DEG/DEAD/STK at k=2)
#
# Runs sequentially (single GPU).
# Run with: bash scripts/experiments/e8_e2_van_ac_eval.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

VAN_AC_EXP="VAN_AC_GoToPosition"
VAN_AC_AGENT="rsl_rl_van_ac_cfg_entry_point"
OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"
INJECT_STEP=100

E8_DIR="${REPO_ROOT}/docs/results/e9_mid_episode"
E2_DIR="${REPO_ROOT}/docs/results/e2_per_mode_isolation"
LOG_FILE="${REPO_ROOT}/docs/results/van_ac_supplementary.log"
mkdir -p "$E8_DIR" "$E2_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo " VAN-MLP-AC supplementary evals          $(date)"
echo " E8: mid-episode + E2: per-mode"
echo " Seeds: ${SEEDS[*]}"
echo "================================================================"

# ── E8: Mid-Episode ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " E8 — VAN-MLP-AC mid-episode injection"
echo "================================================================"
VAN_AC_JSONS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "$VAN_AC_EXP" "$SEED" "$OBS_CKPT_ITER")
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "  [WARN] VAN-MLP-AC checkpoint not found: seed=${SEED}"; continue
    fi
    OUT_DIR="${E8_DIR}/VAN_AC/seed_${SEED}"
    mkdir -p "$OUT_DIR"
    echo "  mid-eval  k=1..4  inject=${INJECT_STEP}  seed=${SEED}"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/eval_mid_episode_failures.py \
        --task="${OBS_TASK}" \
        --agent="${VAN_AC_AGENT}" \
        --checkpoint="${CKPT}" \
        --num_envs=${NUM_EVAL_ENVS} \
        --eval_episodes_per_env=${EVAL_EPISODES} \
        --max_failures=4 \
        --inject_step=${INJECT_STEP} \
        --pos_tol=${POS_TOL} \
        --success_steps=${SUCCESS_STEPS} \
        --output_dir="${OUT_DIR}" \
        --headless
    VAN_AC_JSONS+=("${OUT_DIR}/eval_mid_episode.json")
done

if [ ${#VAN_AC_JSONS[@]} -gt 0 ]; then
    /isaac-sim/kit/python/bin/python3.11 - "VAN-MLP-AC (mid-episode)" "${VAN_AC_JSONS[@]}" << 'PYEOF'
import json, sys, statistics
label, files = sys.argv[1], sys.argv[2:]
all_results = [json.load(open(f))["results"] for f in files if __import__("pathlib").Path(f).exists()]
if not all_results: sys.exit(0)
keys = sorted(all_results[0].keys(), key=int)
print(f"\n  ── {label}  ({len(all_results)} seeds) ──")
print(f"  {'k':>4}  {'SR post-injection':>20}")
print("  " + "-" * 28)
for k in keys:
    vals = [r[k]["success_rate"] for r in all_results if k in r]
    m = statistics.mean(vals)*100; s = statistics.stdev(vals)*100 if len(vals)>1 else 0.0
    print(f"  {k:>4}  {m:>8.1f}±{s:>4.1f}%")
PYEOF
fi

# ── E2: Per-mode generalization ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo " E2 — VAN-MLP-AC per-mode generalization  (k=2)"
echo "================================================================"
for EVAL_MODE in DEG DEAD STK; do
    echo ""
    echo "  ── Eval mode: ${EVAL_MODE} ──"
    SEED_JSONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "$VAN_AC_EXP" "$SEED" "$OBS_CKPT_ITER")
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "    [WARN] not found: seed=${SEED}"; continue
        fi
        OUT_DIR="${E2_DIR}/VAN_AC/eval_${EVAL_MODE}/seed_${SEED}"
        run_eval "$OBS_TASK" "$CKPT" "$OUT_DIR" 2 \
            "--agent=${VAN_AC_AGENT}" \
            "env.task_name=GoToPositionObserver${EVAL_MODE}"
        SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
    done
    aggregate_jsons "VAN-MLP-AC  Eval=${EVAL_MODE}  k=0..2" "${SEED_JSONS[@]}"
done

echo ""
echo "================================================================"
echo " All done.  $(date)"
echo "================================================================"
