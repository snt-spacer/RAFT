#!/usr/bin/env bash
# ============================================================
# Archive all paper checkpoints into paper_checkpoints/
#
# Copies model_ITER.pt for every experiment × seed used in
# the paper into a clean named directory structure.
# Run from repo root: bash scripts/archive_paper_checkpoints.sh
# ============================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(42 1337 7)

OUT="${REPO_ROOT}/docs/paper_checkpoints"
LOGS="${REPO_ROOT}/logs/rsl_rl"

if [ ! -d "$LOGS" ]; then
    echo "[ERROR] logs/rsl_rl not found at ${LOGS}"
    echo "  Run this script inside the Docker container."
    exit 1
fi

echo "================================================================"
echo " Archiving paper checkpoints → ${OUT}"
echo " Source: ${LOGS}"
echo "================================================================"

copy_ckpt() {
    local label="$1" exp="$2" iter="$3"
    local dest="${OUT}/${label}"
    mkdir -p "$dest"
    local found=0
    for SEED in "${SEEDS[@]}"; do
        local run
        run=$(ls -td "${LOGS}/${exp}"/*_seed_${SEED} 2>/dev/null | head -1)
        if [ -z "$run" ]; then
            run=$(ls -td "${LOGS}/${exp}"/*/ 2>/dev/null | head -1)
        fi
        local src="${run}/model_${iter}.pt"
        if [ -f "$src" ]; then
            cp "$src" "${dest}/seed_${SEED}.pt"
            echo "  [OK] ${label}/seed_${SEED}.pt"
            found=$((found+1))
        else
            echo "  [MISS] ${label}/seed_${SEED} — ${src}"
        fi
    done
    [ $found -gt 0 ] || echo "  [WARN] No checkpoints found for ${label}"
}

# ── Primary methods ───────────────────────────────────────────────────────────
copy_ckpt  "RAFT"           "VAN_GRU64_AC_GoToPosition"          4999
copy_ckpt  "VAN_MLP_AC"     "VAN_AC_GoToPosition"                 4999
copy_ckpt  "OBS"            "Observer_ABL_NoMSE"                  4999
copy_ckpt  "OBS_MSE"        "Observer_GoToPosition"               4999
copy_ckpt  "VAN"            "Vanilla_GoToPosition"                2999
copy_ckpt  "GT_ORACLE"      "GT_LongTrain"                        4999

# ── Per-mode specialists (E2) ─────────────────────────────────────────────────
copy_ckpt  "OBS_DEG"        "Observer_DEGOnly_GoToPosition"       4999
copy_ckpt  "OBS_DEAD"       "Observer_DEADOnly_GoToPosition"      4999
copy_ckpt  "OBS_STK"        "Observer_STKOnly_GoToPosition"       4999

# ── E9: RNN ablation (no asymmetric critic) ───────────────────────────────────
copy_ckpt  "RNN_GRU64"      "VAN_GRU64_GoToPosition"              4999
copy_ckpt  "RNN_GRU256"     "VAN_GRU256_GoToPosition"             4999
copy_ckpt  "RNN_LSTM64"     "VAN_LSTM64_GoToPosition"             4999
copy_ckpt  "RNN_LSTM256"    "VAN_LSTM256_GoToPosition"            4999

# ── E10: Asymmetric critic ablation variants ──────────────────────────────────
copy_ckpt  "GRU256_AC"      "VAN_GRU256_AC_GoToPosition"          4999
copy_ckpt  "LSTM64_AC"      "VAN_LSTM64_AC_GoToPosition"          4999
copy_ckpt  "LSTM256_AC"     "VAN_LSTM256_AC_GoToPosition"         4999

# ── E6: Observer extension ablations ─────────────────────────────────────────
copy_ckpt  "ABL_NODETACH"   "Observer_ABL_NoDetach"               4999
copy_ckpt  "ABL_HIST16"     "Observer_ABL_Hist16"                 4999
copy_ckpt  "ABL_HIST128"    "Observer_ABL_Hist128"                4999
copy_ckpt  "ABL_SMALLOBS"   "Observer_ABL_SmallObs"               4999

echo ""
echo "================================================================"
echo " Summary:"
find "$OUT" -name "*.pt" | sort | while read f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  $size  ${f#${OUT}/}"
done
total=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo " Total: ${total}  →  ${OUT}"
echo "================================================================"
