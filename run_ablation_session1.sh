#!/usr/bin/env bash
# Restaurant ablation study -- Kaggle session 1/2 (12h limit).
# Covers: loss balancing (GradNorm vs fixed), loss function (focal/ASL),
# and the lower/middle half of the attention-heads sweep.
# Each run is 3 seeds (42,123,2024); if the session is killed mid-run, the
# already-finished runs above it are unaffected -- just resume with session 2
# and re-launch whichever run in this script never produced a
# multi_seed_summary.json.
set -uo pipefail  # no -e: a single failed/killed run must not stop the rest

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

# --- Loss balancing: does GradNorm actually help vs. fixed weights? ---
run fixedloss    --loss_weighting fixed
run focal        --loss_weighting gradnorm --acd_loss_fn focal
run asl           --loss_weighting gradnorm --acd_loss_fn asl

# --- Attention heads sweep (part 1 of 2; default=8 already run as baseline) ---
run heads2        --loss_weighting gradnorm --num_attention_heads 2
run heads4        --loss_weighting gradnorm --num_attention_heads 4
run heads12       --loss_weighting gradnorm --num_attention_heads 12
run heads16       --loss_weighting gradnorm --num_attention_heads 16

echo "[$(date '+%H:%M:%S')] Session 1 finished."
