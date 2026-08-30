#!/usr/bin/env bash
# Restaurant ablation study -- Kaggle session 2/2 (12h limit).
# Covers: the remaining attention-heads sweep point, plus the full
# adapter-dimension sweep.
# Run this after run_ablation_session1.sh (or independently -- each run
# checks for an existing multi_seed_summary.json and skips if already done,
# so re-running either script is always safe).
set -uo pipefail

TRAIN=Res_ABSA/Train.txt
DEV=Res_ABSA/Dev.txt
TEST=Res_ABSA/Test.txt
MODEL=vinai/phobert-base-v2
SEEDS="42,123,2024"

run() {
    local name="$1"; shift
    local out="outputs/phobert_mtl_acsa_restaurant_v2_${name}"
    if [ -f "${out}/multi_seed_summary.json" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP ${name} (multi_seed_summary.json already exists)"
        return
    fi
    echo "[$(date '+%H:%M:%S')] START ${name}"
    SECONDS=0
    python3 train_mtl_acsa_v2.py \
        --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
        --model_name "$MODEL" --seeds "$SEEDS" \
        --output_dir "$out" "$@"
    echo "[$(date '+%H:%M:%S')] DONE  ${name} (${SECONDS}s)"
}

# --- Attention heads sweep (part 2 of 2) ---
run heads24       --loss_weighting gradnorm --num_attention_heads 24

# --- Adapter dimension / hidden-size sweep (default=192 already run as baseline) ---
run adapter64     --loss_weighting gradnorm --adapter_dim 64
run adapter96     --loss_weighting gradnorm --adapter_dim 96
run adapter256    --loss_weighting gradnorm --adapter_dim 256
run adapter384    --loss_weighting gradnorm --adapter_dim 384
run adapter512    --loss_weighting gradnorm --adapter_dim 512

echo "[$(date '+%H:%M:%S')] Session 2 finished."
