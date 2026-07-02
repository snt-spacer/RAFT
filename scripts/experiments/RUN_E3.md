# E3 Severity Sweep — Re-run Commands

The previous results were produced with a buggy `_run_episodes` (no warmup step,
no pre-step position snapshot, non-latched success). Run the commands below
inside the container to overwrite them with correct results.

---

## 0. Confirm checkpoints exist

```bash
# Run from repo root inside the container
for SEED in 42 1337 7; do
    CKPT=$(ls -td logs/rsl_rl/Observer_GoToPosition/*_seed_${SEED} 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -td logs/rsl_rl/Observer_GoToPosition/*/ 2>/dev/null | head -1)
    fi
    echo "seed=${SEED}  →  ${CKPT}/model_4999.pt"
done
```

If any path is missing, substitute the correct checkpoint path in the commands below.

---

## 1. DEG sweep (continuous degradation)

```bash
PYTHON=/isaac-sim/kit/python/bin/python3.11

for SEED in 42 1337 7; do
    CKPT_DIR=$(ls -td logs/rsl_rl/Observer_GoToPosition/*_seed_${SEED} 2>/dev/null | head -1)
    if [ -z "$CKPT_DIR" ]; then
        CKPT_DIR=$(ls -td logs/rsl_rl/Observer_GoToPosition/*/ 2>/dev/null | head -1)
    fi
    CKPT="${CKPT_DIR}/model_4999.pt"

    echo ""
    echo "=== DEG  seed=${SEED}  ckpt=${CKPT} ==="
    python3 scripts/experiments/eval_severity.py \
        --task Isaaclab-RANSv2-Observer-Position-v0 \
        --checkpoint "${CKPT}" \
        --mode deg \
        --severities 1.0 0.9 0.7 0.5 0.3 0.1 0.0 \
        --output_dir "docs/results/e3_severity_sweep/deg/seed_${SEED}" \
        --seed "${SEED}" \
        --num_envs 512 \
        --eval_episodes_per_env 10 \
        --pos_tol 0.05 \
        --success_steps 50 \
        --headless
done
```

---

## 2. STK sweep (stuck-on)

```bash
PYTHON=/isaac-sim/kit/python/bin/python3.11

for SEED in 42 1337 7; do
    CKPT_DIR=$(ls -td logs/rsl_rl/Observer_GoToPosition/*_seed_${SEED} 2>/dev/null | head -1)
    if [ -z "$CKPT_DIR" ]; then
        CKPT_DIR=$(ls -td logs/rsl_rl/Observer_GoToPosition/*/ 2>/dev/null | head -1)
    fi
    CKPT="${CKPT_DIR}/model_4999.pt"

    echo ""
    echo "=== STK  seed=${SEED}  ckpt=${CKPT} ==="
    python3 scripts/experiments/eval_severity.py \
        --task Isaaclab-RANSv2-Observer-Position-v0 \
        --checkpoint "${CKPT}" \
        --mode stk \
        --severities 0.0 0.1 0.2 0.4 0.6 0.8 1.0 \
        --output_dir "docs/results/e3_severity_sweep/stk/seed_${SEED}" \
        --seed "${SEED}" \
        --num_envs 512 \
        --eval_episodes_per_env 10 \
        --pos_tol 0.05 \
        --success_steps 50 \
        --headless
done
```

---

## 3. Quick sanity check after runs complete

```bash
/isaac-sim/kit/python/bin/python3.11 - << 'EOF'
import json

for mode in ["deg", "stk"]:
    print(f"\n{'='*50}")
    print(f"  {mode.upper()} results")
    print(f"{'='*50}")
    for seed in [42, 1337, 7]:
        path = f"docs/results/e3_severity_sweep/{mode}/seed_{seed}/severity_results.json"
        try:
            d = json.load(open(path))
            srs = {k: f"{v['success_rate']*100:.0f}%" for k, v in sorted(d.items(), key=lambda x: float(x[0]))}
            print(f"  seed={seed}: {srs}")
        except FileNotFoundError:
            print(f"  seed={seed}: MISSING")
EOF
```

Expected: near-healthy severities (DEG=1.0, STK=0.0) should approach
the E1 k=1 success rate of ~100%. Rates should degrade as severity
increases (more severe failure).
