#!/usr/bin/env bash
# ============================================================
# E8 — Mid-Episode Failure Injection for RAFT (GRU-64-AC)
#
# Adds RAFT results to docs/results/e9_mid_episode/RAFT/
# so they can be overlaid on the existing E8 figure alongside
# OBS (λ=0), GT_LONGTR, and VAN.
#
# Protocol:
#   - Episodes start healthy (k=0)
#   - At inject_step=100, k thrusters are suddenly degraded
#   - Post-injection SR is measured; compares RAFT adaptation
#     ability to OBS (history-buffer observer) and the oracle
#
# Run with: bash scripts/experiments/e8_raft_mid_episode.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RESULTS_DIR="${REPO_ROOT}/docs/results/e9_mid_episode"
RESULTS_FILE="${RESULTS_DIR}/raft_summary.txt"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_FILE") 2>&1

RAFT_TASK="Isaaclab-RANSv2-Observer-Position-v0"
RAFT_AGENT="rsl_rl_rnn_gru64_ac_cfg_entry_point"
INJECT_STEP=100

echo "================================================================"
echo " E8 — Mid-Episode Failure Injection: RAFT    $(date)"
echo ""
echo " task        : ${RAFT_TASK}"
echo " agent       : ${RAFT_AGENT}"
echo " inject_step : ${INJECT_STEP}"
echo " eval_envs   : ${NUM_EVAL_ENVS}"
echo " episodes/env: ${EVAL_EPISODES}"
echo " seeds       : ${SEEDS[*]}"
echo "================================================================"

run_mid_eval() {
    local task="$1" agent="$2" ckpt="$3" out_dir="$4" max_k="$5"
    mkdir -p "$out_dir"
    echo "  mid-eval  task=${task}  k=1..${max_k}  inject=${INJECT_STEP}  ckpt=$(basename "$ckpt")"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/eval_mid_episode_failures.py \
        --task="${task}" \
        --agent="${agent}" \
        --checkpoint="${ckpt}" \
        --num_envs=${NUM_EVAL_ENVS} \
        --eval_episodes_per_env=${EVAL_EPISODES} \
        --max_failures="${max_k}" \
        --inject_step=${INJECT_STEP} \
        --pos_tol=${POS_TOL} \
        --success_steps=${SUCCESS_STEPS} \
        --output_dir="${out_dir}" \
        --headless
}

# ── RAFT (GRU-64-AC) ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " RAFT — GRU-64-AC with asymmetric critic (proposed)"
echo "================================================================"
RAFT_JSONS=()
for SEED in "${SEEDS[@]}"; do
    CKPT=$(find_checkpoint "VAN_GRU64_AC_GoToPosition" "$SEED" "$OBS_CKPT_ITER")
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "  [WARN] RAFT checkpoint not found: seed=${SEED}"
        continue
    fi
    OUT_DIR="${RESULTS_DIR}/RAFT/seed_${SEED}"
    run_mid_eval "$RAFT_TASK" "$RAFT_AGENT" "$CKPT" "$OUT_DIR" 4
    RAFT_JSONS+=("${OUT_DIR}/eval_mid_episode.json")
done

if [ ${#RAFT_JSONS[@]} -gt 0 ]; then
    /isaac-sim/kit/python/bin/python3.11 - "RAFT (GRU-64-AC, proposed)" "${RAFT_JSONS[@]}" << 'PYEOF'
import json, sys, statistics

label = sys.argv[1]
files = sys.argv[2:]
all_results = []
for f in files:
    try:
        with open(f) as fp:
            d = json.load(fp)
            all_results.append(d["results"])
    except Exception as e:
        print(f"  [WARN] {f}: {e}")

if not all_results:
    print(f"  [WARN] No data for {label}")
    sys.exit(0)

keys = sorted(all_results[0].keys(), key=lambda x: int(x))

print(f"\n  ── {label}  ({len(all_results)} seeds) ──")
print(f"  {'k':>4}  {'SR post-injection':>20}")
print("  " + "-" * 28)

for k in keys:
    sr_vals = [r[k]["success_rate"] for r in all_results if k in r]
    sr_mean = statistics.mean(sr_vals) if sr_vals else float("nan")
    sr_std  = statistics.stdev(sr_vals) if len(sr_vals) > 1 else 0.0
    print(f"  {k:>4}  {sr_mean*100:>8.1f}±{sr_std*100:>4.1f}%")
PYEOF
fi

echo ""
echo "================================================================"
echo " Done. Results: ${RESULTS_DIR}/RAFT/"
echo "================================================================"
