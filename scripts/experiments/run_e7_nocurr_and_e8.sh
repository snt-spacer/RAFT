#!/usr/bin/env bash
# ============================================================
# Combined: GT-NOCURR (E7) + Observer Interpretability (E8)
#
# Steps:
#   1. Clear stale robots_cfg pycache so CuboThrusterFailureNoCurr loads
#   2. Train GT-NOCURR (3 seeds, 2000 iters, k=4 from step 1)
#   3. Evaluate GT-NOCURR k=0..4
#   4. Run E8 probe on OBS_FULL and ABL_NOMSE checkpoints
#
# Run with: bash scripts/experiments/run_e7_nocurr_and_e8.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

E7_RESULTS_DIR="${REPO_ROOT}/docs/results/e7_gt_collapse"
E8_RESULTS_DIR="${REPO_ROOT}/docs/results/e8_observer_probe"
RESULTS_FILE="${E7_RESULTS_DIR}/summary_nocurr_e8.txt"
mkdir -p "$E7_RESULTS_DIR" "$E8_RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

GT_TASK="Isaaclab-RANSv2-Observer-Position-v0_GT"

echo "================================================================"
echo " GT-NOCURR + E8 Observer Probe          $(date)"
echo "================================================================"

# ── Step 1: Clear stale pycache so NoCurr config is picked up ─────────────────
echo ""
echo ">>> Clearing stale robots_cfg pycache ..."
find "${REPO_ROOT}/source" -path "*/robots_cfg/__pycache__/__init__*.pyc" -delete 2>/dev/null || true
echo ">>> Done."

# ── Step 2: GT-NOCURR training ────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " GT-NOCURR Training (2000 iters, k=4 from step 1)"
echo "================================================================"

GT_TASK="Isaaclab-RANSv2-GroundTruth-Position-v0"

for SEED in "${SEEDS[@]}"; do
    if ! run_train "GT-NOCURR" "$GT_TASK" "rsl_rl_cfg_entry_point" "$SEED" \
        "env.robot_name=CuboThrusterFailureNoCurr" \
        "agent.experiment_name=GT_NoCurr"; then
        echo "[WARN] Training failed: GT-NOCURR seed=${SEED} — continuing"
    fi
done

# ── Step 3: GT-NOCURR evaluation ──────────────────────────────────────────────
echo ""
echo "================================================================"
echo " GT-NOCURR Evaluation (k=0..4, mixed modes)"
echo "================================================================"

SEED_JSONS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "GT_NoCurr" "$SEED" $GT_CKPT_ITER)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "  [WARN] Checkpoint not found: GT-NOCURR seed=${SEED}"
        continue
    fi
    OUT_DIR="${E7_RESULTS_DIR}/GT_NOCURR/seed_${SEED}"
    run_eval "$GT_TASK" "$CKPT" "$OUT_DIR" 4
    SEED_JSONS+=("${OUT_DIR}/eval_gt_failures.json")
done

if [ ${#SEED_JSONS[@]} -gt 0 ]; then
    aggregate_jsons "GT_NOCURR  k=0..4  mixed-modes" "${SEED_JSONS[@]}"
fi

# Compare all three GT variants
echo ""
echo "──────────────────────  GT-ORIGINAL (from E1, reference)  ──────────────────────"
ORIG_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${REPO_ROOT}/docs/results/e1_main_comparison/GT/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && ORIG_JSONS+=("$F")
done
[ ${#ORIG_JSONS[@]} -gt 0 ] && aggregate_jsons "GT-ORIGINAL (2000 iters, k_max≤2)" "${ORIG_JSONS[@]}"

echo ""
echo "──────────────────────  GT-LONGTR (from E7)  ──────────────────────"
LT_JSONS=()
for SEED in "${SEEDS[@]}"; do
    F="${E7_RESULTS_DIR}/GT_LONGTR/seed_${SEED}/eval_gt_failures.json"
    [ -f "$F" ] && LT_JSONS+=("$F")
done
[ ${#LT_JSONS[@]} -gt 0 ] && aggregate_jsons "GT-LONGTR (5000 iters, curriculum)" "${LT_JSONS[@]}"

# ── Step 4: E8 Observer Interpretability Probe ────────────────────────────────
echo ""
echo "================================================================"
echo " E8 — Observer Interpretability Probe"
echo "================================================================"

OBS_TASK="Isaaclab-RANSv2-Observer-Position-v0"

declare -A PROBE_EXP
PROBE_EXP[OBS_FULL]="Observer_GoToPosition"
PROBE_EXP[ABL_NOMSE]="Observer_ABL_NoMSE"

for VARIANT in OBS_FULL ABL_NOMSE; do
    echo ""
    echo "──────────────────────  ${VARIANT}  ──────────────────────"
    for SEED in "${SEEDS[@]}"; do
        CKPT=$(find_checkpoint "${PROBE_EXP[$VARIANT]}" "$SEED" $OBS_CKPT_ITER)
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [WARN] Checkpoint not found: ${VARIANT} seed=${SEED}"
            continue
        fi
        OUT_DIR="${E8_RESULTS_DIR}/${VARIANT}/seed_${SEED}"
        mkdir -p "$OUT_DIR"
        echo "  Probing ${VARIANT} seed=${SEED} ..."
        cd "$REPO_ROOT"
        "$PYTHON_EXE" scripts/rsl_rl/eval_observer_probe.py \
            --task="${OBS_TASK}" \
            --checkpoint="${CKPT}" \
            --variant="${VARIANT}" \
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

# ── E8 Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " E8 RESULTS SUMMARY"
echo "================================================================"

/isaac-sim/kit/python/bin/python3.11 - "${E8_RESULTS_DIR}" << 'PYEOF'
import json, pathlib, sys, statistics

base = pathlib.Path(sys.argv[1])
for variant in ["OBS_FULL", "ABL_NOMSE"]:
    print(f"\n  ── {variant} ──")
    probes = list((base / variant).glob("seed_*/probe_results.json"))
    if not probes:
        print("    [no results]")
        continue

    r2_k2_vals, r2_k0_vals = [], []
    grad_ratio = {0: [], 2: [], 4: []}
    zero_delta = {k: [] for k in range(5)}

    for p in sorted(probes):
        d = json.load(open(p))
        r2_k2_vals.append(d["linear_probe_k2"]["r2_overall"])
        r2_k0_vals.append(d["linear_probe_k0"]["r2_overall"])
        for k in [0, 2, 4]:
            sk = str(k)
            ga = d.get("gradient_attribution", {})
            if sk in ga:
                ratio = ga[sk].get("ratio_dhat_otask", float("nan"))
                if ratio == ratio:
                    grad_ratio[k].append(ratio)
        for k in range(5):
            sk = str(k)
            zt = d.get("zeroing_test", {})
            if sk in zt:
                zero_delta[k].append(zt[sk]["delta_sr"])

    def fmt(vals):
        if not vals: return "n/a"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        return f"{m:.3f}±{s:.3f}"

    print(f"    Linear probe R² (k=0): {fmt(r2_k0_vals)}")
    print(f"    Linear probe R² (k=2): {fmt(r2_k2_vals)}")
    print(f"    Gradient ratio ||∂μ/∂D_hat|| / ||∂μ/∂o_task||:")
    for k in [0, 2, 4]:
        print(f"      k={k}: {fmt(grad_ratio[k])}")
    print(f"    Zeroing ΔSR (normal − zeroed, pp):")
    for k in range(5):
        deltas = zero_delta[k]
        if deltas:
            m = statistics.mean(deltas) * 100
            verdict = "MATTERS" if abs(m) > 2 else "irrelevant"
            print(f"      k={k}: {m:+.1f}pp  [{verdict}]")
PYEOF

echo ""
echo "Done. Full log: ${RESULTS_FILE}"
