#!/usr/bin/env bash
# Education baselines for Table 4 -- Kaggle session (GPU). Fills the rows of
# Table 4 that are currently "--" for Education: BERT-based (PhoBERT / XLM-R
# / Ensemble BERTs) and the 4 instruction-tuning variants. Education's
# Statistics row (SVM/CNN/BiLSTM-CNN) already has real published numbers
# from the dataset's own paper (TNUJST5101), so this script does not
# re-run CNN or SVM for Education -- only baselines/run_baselines_beauty.sh
# needs those, since no prior baseline of any kind exists for Beauty.
#
# Idempotent: each run is skipped if its summary file already exists, so
# re-running this script (e.g. after a Kaggle session gets cut off) is safe.
set -uo pipefail

TRAIN=Education_ABSA/Train.txt
DEV=Education_ABSA/Dev.txt
TEST=Education_ABSA/Test.txt
DOMAIN_NAME="university course evaluation"   # used inside T5 prompt text

skip_if_exists() {
    [ -f "$1" ] && { echo "[$(date '+%H:%M:%S')] SKIP (already exists: $1)"; return 0; }
    return 1
}

echo "[$(date '+%H:%M:%S')] === BERT-based: PhoBERT ==="
out=outputs/bert_education_phobert
skip_if_exists "$out/multi_seed_summary.json" || \
python3 baselines/bert_baseline.py --mode train --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$out" --model_name vinai/phobert-base-v2 --segmenter pyvi --seeds 42,123,2024

echo "[$(date '+%H:%M:%S')] === BERT-based: XLM-R ==="
out=outputs/bert_education_xlmr
skip_if_exists "$out/multi_seed_summary.json" || \
python3 baselines/bert_baseline.py --mode train --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$out" --model_name xlm-roberta-base --segmenter none --seeds 42,123,2024

echo "[$(date '+%H:%M:%S')] === BERT-based: Ensemble BERTs ==="
out=outputs/bert_education_ensemble
skip_if_exists "$out/test_metrics.json" || \
python3 baselines/bert_baseline.py --mode ensemble --test_path "$TEST" --output_dir "$out" \
    --checkpoints outputs/bert_education_phobert/best_model.pt outputs/bert_education_xlmr/best_model.pt

echo "[$(date '+%H:%M:%S')] === Instruction tuning: 4 variants ==="
for format in code nl; do
    for lang in vi en; do
        out="outputs/t5_education_${format}_${lang}"
        skip_if_exists "$out/multi_seed_summary.json" || \
        python3 baselines/t5_instruction_tuning.py --domain "$DOMAIN_NAME" --format "$format" --lang "$lang" \
            --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" --output_dir "$out" --seeds 42,123,2024
    done
done

echo "[$(date '+%H:%M:%S')] Education baselines finished."
