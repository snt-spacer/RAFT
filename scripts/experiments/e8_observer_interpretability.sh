#!/usr/bin/env bash
# ============================================================
# E8 — Observer Interpretability
#
# Runs three analyses on OBS_FULL and ABL_NOMSE checkpoints:
#   1. Linear probe (D_hat → D_gt, R²)
#   2. Gradient attribution (||∂μ/∂D_hat|| vs ||∂μ/∂o_task||)
#   3. D_hat zeroing test (ΔSR when D_hat is forced to zeros)
#
# Run with: bash scripts/experiments/e8_observer_interpretability.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e8_observer_probe"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

TASK="Isaaclab-RANSv2-Observer-Position-v0"

echo "================================================================"
echo " E8 — Observer Interpretability                $(date)"
echo " Analyses: linear probe, gradient attribution, zeroing test"
echo " Variants: OBS_FULL, ABL_NOMSE"
echo "================================================================"

declare -A PROBE_EXP PROBE_VARIANT
PROBE_EXP[OBS_FULL]="Observer_GoToPosition"
PROBE_EXP[ABL_NOMSE]="Observer_ABL_NoMSE"
PROBE_VARIANT[OBS_FULL]="OBS_FULL"
PROBE_VARIANT[ABL_NOMSE]="ABL_NOMSE"

for VARIANT in OBS_FULL ABL_NOMSE; do
    echo ""
    echo "──────────────────────  ${VARIANT}  ──────────────────────"

    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${PROBE_EXP[$VARIANT]}" "$SEED" $OBS_CKPT_ITER)
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [WARN] Checkpoint not found: ${VARIANT} seed=${SEED}"
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/${VARIANT}/seed_${SEED}"
        mkdir -p "$OUT_DIR"
        echo ""
        echo "  Probing ${VARIANT} seed=${SEED} ..."
        echo "  Checkpoint: ${CKPT}"
        cd "$REPO_ROOT"
        "$PYTHON_EXE" scripts/rsl_rl/eval_observer_probe.py \
            --task="${TASK}" \
            --checkpoint="${CKPT}" \
            --variant="${PROBE_VARIANT[$VARIANT]}" \
            --num_envs=${NUM_EVAL_ENVS} \
            --eval_episodes_per_env=${EVAL_EPISODES} \
            --max_failures=4 \
            --probe_steps=3000 \
            --pos_tol=${POS_TOL} \
            --success_steps=${SUCCESS_STEPS} \
            --output_dir="${OUT_DIR}" \
            --headless
    done
done

# ── Aggregate and print summary ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo " RESULTS SUMMARY"
echo "================================================================"

"$PYTHON_EXE" - "${RESULTS_DIR}" << 'PYEOF'
import json, pathlib, sys, statistics

base = pathlib.Path(sys.argv[1])

for variant in ["OBS_FULL", "ABL_NOMSE"]:
    print(f"\n  ── {variant} ──")
    probe_jsons = list((base / variant).glob("seed_*/probe_results.json"))
    if not probe_jsons:
        print("    [no results]")
        continue

    all_r2 = []
    all_zero_delta = {k: [] for k in range(5)}

    for path in sorted(probe_jsons):
        d = json.load(open(path))
        all_r2.append(d["linear_probe_k2"]["r2_overall"])
        for k in range(5):
            sk = str(k)
            if "zeroing_test" in d and sk in d["zeroing_test"]:
                all_zero_delta[k].append(d["zeroing_test"][sk]["delta_sr"])

    print(f"    Linear probe R² (k=2): mean={statistics.mean(all_r2):.3f}  "
          f"seeds={len(all_r2)}")
    print(f"    Zeroing ΔSR (normal - zeroed):")
    for k in range(5):
        deltas = all_zero_delta[k]
        if deltas:
            print(f"      k={k}: {statistics.mean(deltas)*100:+.1f}pp  "
                  f"(zeroing {'matters' if abs(statistics.mean(deltas)) > 0.02 else 'irrelevant'})")
PYEOF

echo ""
echo "Done. Results: ${RESULTS_FILE}"
