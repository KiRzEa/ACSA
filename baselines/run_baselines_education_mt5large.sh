#!/usr/bin/env bash
# Education: mT5-large only (both GPUs, model-parallel via --device_map_auto).
# Split into its own session/script -- mT5-large and viT5-large back-to-back
# in one session risk one crash (OOM, disk, session timeout) taking out
# progress on both; running each alone means a failure only costs itself.
# See run_baselines_education_vit5large.sh for the other one.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Kaggle's default transformers build fails to load Salesforce/codet5-base's
# tokenizer (TypeError: extra_special_tokens must be a list/tuple ...) --
# pin to a version confirmed working against every model used in this suite.
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate

TRAIN=Education_ABSA/Train.txt
DEV=Education_ABSA/Dev.txt
TEST=Education_ABSA/Test.txt
OUT=outputs/mt5large_education

if [ -f "$OUT/multi_seed_summary.json" ]; then
    echo "[$(date '+%H:%M:%S')] SKIP (already exists: $OUT/multi_seed_summary.json)"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] === T5-based: mT5-large (both GPUs, model-parallel) ==="
# mT5-large (~1.2B params) doesn't fit on a single ~15GB Kaggle GPU at all --
# AdamW's optimizer state alone already exceeds 15GB before any activations,
# so batch_size=1 doesn't save it. --device_map_auto splits the model's
# layers across BOTH GPUs (real model parallelism) -- do NOT pin
# CUDA_VISIBLE_DEVICES to a single GPU, it needs both visible.
python3 baselines/t5_seq2seq_baseline.py --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$OUT" --model_name google/mt5-large --seeds 42 \
    --batch_size 2 --eval_batch_size 4 --gradient_checkpointing --device_map_auto

# Clear the HF model cache -- mT5-large alone caches ~10GB (both .bin and
# .safetensors); leaving it around is what exhausted Kaggle's disk
# ("No space left on device") in an earlier run.
rm -rf ~/.cache/huggingface/hub 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] Education mT5-large finished."
