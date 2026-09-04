#!/usr/bin/env bash
# Beauty baselines for Table 4 -- Kaggle session (GPU). Fills every row of
# Table 4 that is currently "--" for Beauty: CNN, BERT-based (PhoBERT / XLM-R
# / Ensemble BERTs), and the 4 instruction-tuning variants. The Statistics
# row (SVM) is CPU-only and fast enough to run locally -- see
# baselines/svm_tfidf_baseline.py, no Kaggle session needed for it.
#
# Idempotent: each run is skipped if its summary file already exists, so
# re-running this script (e.g. after a Kaggle session gets cut off) is safe.
set -uo pipefail

# Kaggle's default transformers build fails to load Salesforce/codet5-base's
# tokenizer (TypeError: extra_special_tokens must be a list/tuple ...) --
# pin to a version confirmed working against every model used below.
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate

TRAIN=Beauty_ABSA/Train.txt
DEV=Beauty_ABSA/Dev.txt
TEST=Beauty_ABSA/Test.txt
DOMAIN_NAME="beauty product"   # used inside T5 prompt text

skip_if_exists() {
    [ -f "$1" ] && { echo "[$(date '+%H:%M:%S')] SKIP (already exists: $1)"; return 0; }
    return 1
}

echo "[$(date '+%H:%M:%S')] === CNN ==="
out=outputs/cnn_beauty
skip_if_exists "$out/multi_seed_summary.json" || \
python3 baselines/cnn_baseline.py --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$out" --seeds 42,123,2024

echo "[$(date '+%H:%M:%S')] === BERT-based: PhoBERT ==="
out=outputs/bert_beauty_phobert
skip_if_exists "$out/multi_seed_summary.json" || \
python3 baselines/bert_baseline.py --mode train --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$out" --model_name vinai/phobert-base-v2 --segmenter pyvi --seeds 42,123,2024

echo "[$(date '+%H:%M:%S')] === BERT-based: XLM-R ==="
out=outputs/bert_beauty_xlmr
skip_if_exists "$out/multi_seed_summary.json" || \
python3 baselines/bert_baseline.py --mode train --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$out" --model_name xlm-roberta-base --segmenter none --seeds 42,123,2024

echo "[$(date '+%H:%M:%S')] === BERT-based: Ensemble BERTs ==="
out=outputs/bert_beauty_ensemble
skip_if_exists "$out/test_metrics.json" || \
python3 baselines/bert_baseline.py --mode ensemble --test_path "$TEST" --output_dir "$out" \
    --checkpoints outputs/bert_beauty_phobert/best_model.pt outputs/bert_beauty_xlmr/best_model.pt

echo "[$(date '+%H:%M:%S')] === Instruction tuning: 4 variants ==="
for format in code nl; do
    for lang in vi en; do
        out="outputs/t5_beauty_${format}_${lang}"
        skip_if_exists "$out/multi_seed_summary.json" || \
        python3 baselines/t5_instruction_tuning.py --domain "$DOMAIN_NAME" --format "$format" --lang "$lang" \
            --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" --output_dir "$out" --seeds 42,123,2024
    done
done

echo "[$(date '+%H:%M:%S')] Beauty baselines finished."
