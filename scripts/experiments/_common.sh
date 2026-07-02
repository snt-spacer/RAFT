#!/usr/bin/env bash
# Shared helpers sourced by all experiment scripts.
# Source with: source "$(dirname "$0")/_common.sh"
#
# Caller scripts must set up self-logging BEFORE calling any helper:
#   mkdir -p "$RESULTS_DIR"
#   exec > >(tee -a "$RESULTS_FILE") 2>&1

PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEEDS=(42 1337 7)
NUM_TRAIN_ENVS=4096
NUM_EVAL_ENVS=512
EVAL_EPISODES=10
POS_TOL=0.05
SUCCESS_STEPS=50

# Checkpoint iteration (0-indexed last iter) per agent cfg
OBS_CKPT_ITER=4999   # Observer:     max_iterations=5000
HIS_CKPT_ITER=2999   # Transformer:  max_iterations=3000
GT_CKPT_ITER=1999    # GroundTruth:  max_iterations=2000

# ── Helpers ──────────────────────────────────────────────────────────────────

# find_checkpoint EXP_NAME SEED CKPT_ITER
# Returns path to model_<CKPT_ITER>.pt inside the most recent seed-<SEED> run.
find_checkpoint() {
    local exp_name="$1" seed="$2" ckpt_iter="$3"
    local log_dir="${REPO_ROOT}/logs/rsl_rl/${exp_name}"
    local run_dir
    run_dir=$(ls -td "${log_dir}"/*_seed_${seed} 2>/dev/null | head -1)
    if [ -z "$run_dir" ]; then
        run_dir=$(ls -td "${log_dir}"/*/  2>/dev/null | head -1)
    fi
    if [ -z "$run_dir" ]; then
        echo ""
        return
    fi
    echo "${run_dir}/model_${ckpt_iter}.pt"
}

# run_train LABEL TASK AGENT SEED EXTRA_ARGS...
run_train() {
    local label="$1" task="$2" agent="$3" seed="$4"
    shift 4
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  TRAIN  ${label}  seed=${seed}"
    echo "════════════════════════════════════════════════════════════"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/train.py \
        --task="${task}" \
        --agent="${agent}" \
        --num_envs=${NUM_TRAIN_ENVS} \
        --headless \
        --seed="${seed}" \
        "$@"
}

# run_eval TASK CHECKPOINT OUTPUT_DIR MAX_FAILURES EXTRA_ARGS...
# Output flows to stdout (captured by exec tee in caller).
run_eval() {
    local task="$1" ckpt="$2" out_dir="$3" max_k="$4"
    shift 4
    mkdir -p "$out_dir"
    echo "  eval  task=${task}  k=0..${max_k}  ckpt=$(basename "$ckpt")"
    cd "$REPO_ROOT"
    "$PYTHON_EXE" scripts/rsl_rl/eval_gt_failures.py \
        --task="${task}" \
        --checkpoint="${ckpt}" \
        --num_envs=${NUM_EVAL_ENVS} \
        --eval_episodes_per_env=${EVAL_EPISODES} \
        --max_failures="${max_k}" \
        --pos_tol=${POS_TOL} \
        --success_steps=${SUCCESS_STEPS} \
        --output_dir="${out_dir}" \
        --headless \
        "$@"
}

# print_json JSON_FILE LABEL
print_json() {
    local json_file="$1" label="$2"
    if [ ! -f "$json_file" ]; then
        echo "  [WARN] No results file at ${json_file}"
        return
    fi
    /isaac-sim/kit/python/bin/python3.11 - "$json_file" "$label" << 'PYEOF'
import json, sys
fname, label = sys.argv[1], sys.argv[2]
with open(fname) as f:
    data = json.load(f)

print(f"\n  ── {label} ──")
try:
    keys = sorted(data.keys(), key=lambda x: float(x))
except ValueError:
    keys = sorted(data.keys())

sample = next((v for v in data.values() if isinstance(v, dict)), None)
if sample is None:
    print(json.dumps(data, indent=2))
    sys.exit(0)

fields = [k for k, v in sample.items() if isinstance(v, (int, float))][:5]
hdr = f"  {'key':>8}" + "".join(f"  {f[:14]:>14}" for f in fields)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for k in keys:
    row = f"  {k:>8}"
    if k in data and isinstance(data[k], dict):
        for field in fields:
            val = data[k].get(field, float("nan"))
            if field == "success_rate":
                row += f"  {val*100:>13.1f}%"
            elif "pos_m" in field:
                row += f"  {val*100:>13.2f}cm"
            else:
                row += f"  {val:>14.4f}"
    print(row)
PYEOF
}

# aggregate_jsons LABEL JSON_FILE [JSON_FILE...]
# JSON format: {"results": {"0": {"success_rate": ..., ...}, "1": {...}, ...}, ...}
aggregate_jsons() {
    local label="$1"
    shift
    local json_files=("$@")
    /isaac-sim/kit/python/bin/python3.11 - "$label" "${json_files[@]}" << 'PYEOF'
import json, sys, statistics, math

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
print(f"  {'k':>4}  {'success_rate':>16}  {'final_pos_cm':>16}")
print("  " + "-" * 42)

for k in keys:
    sr_vals = [r[k]["success_rate"] for r in all_results if k in r]
    fp_vals = [
        r[k]["final_pos_success_m"]["mean"] * 100
        for r in all_results
        if k in r and not math.isnan(r[k]["final_pos_success_m"]["mean"])
    ]
    sr_mean = statistics.mean(sr_vals) if sr_vals else float("nan")
    sr_std  = statistics.stdev(sr_vals) if len(sr_vals) > 1 else 0.0
    fp_mean = statistics.mean(fp_vals) if fp_vals else float("nan")
    fp_std  = statistics.stdev(fp_vals) if len(fp_vals) > 1 else 0.0
    fp_str  = f"{fp_mean:>6.2f}±{fp_std:.2f}cm" if fp_vals else "     n/a"
    print(f"  {k:>4}  {sr_mean*100:>7.1f}±{sr_std*100:>5.1f}%{'':>2}  {fp_str:>16}")
PYEOF
}
