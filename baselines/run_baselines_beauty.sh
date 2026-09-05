#!/usr/bin/env bash
# Beauty baselines for Table 4 -- remainder only. Everything else for Beauty
# is done and already recorded in the paper: SVM, CNN, BiLSTM-CNN, BERT-based
# (PhoBERT/XLM-R/Ensemble BERTs), viT5-base, and 3 of the 4 instruction-tuning
# variants (Code-Vi/Code-En/NL-Vi). mT5-large and viT5-large each run in
# their own dedicated script now (run_baselines_beauty_mt5large.sh /
# run_baselines_beauty_vit5large.sh) -- run those as separate Kaggle
# sessions, not alongside this one or each other, so a crash in one doesn't
# cost progress on the other.
#
# Idempotent: skipped if its summary file already exists, so re-running this
# script is safe WITHIN the same session (a brand new session starts with an
# empty outputs/, so this can't detect runs completed in a previous session
# -- tell me the results instead and I'll record them / trim the script).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Kaggle's default transformers build fails to load Salesforce/codet5-base's
# tokenizer (TypeError: extra_special_tokens must be a list/tuple ...) --
# pin to a version confirmed working against every model used in this suite.
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate

TRAIN=Beauty_ABSA/Train.txt
DEV=Beauty_ABSA/Dev.txt
TEST=Beauty_ABSA/Test.txt
DOMAIN_NAME="beauty product"   # used inside T5 prompt text
OUT=outputs/t5_beauty_nl_en

if [ -f "$OUT/multi_seed_summary.json" ]; then
    echo "[$(date '+%H:%M:%S')] SKIP (already exists: $OUT/multi_seed_summary.json)"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] === Instruction tuning: NL-En (the one Beauty variant still missing) ==="
python3 baselines/t5_instruction_tuning.py --domain "$DOMAIN_NAME" --format nl --lang en \
    --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" --output_dir "$OUT" --seeds 42
rm -rf ~/.cache/huggingface/hub 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] Beauty baselines finished."
