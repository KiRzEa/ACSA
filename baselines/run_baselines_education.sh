#!/usr/bin/env bash
# Education baselines for Table 4 -- Kaggle session (GPU). Fills the rows of
# Table 4 that are currently "--" for Education: BERT-based (PhoBERT / XLM-R
# / Ensemble BERTs), T5-based (mT5-large / viT5-large / viT5-base), and the
# 4 instruction-tuning variants. Education's Statistics row (SVM/CNN/
# BiLSTM-CNN) already has real published numbers from the dataset's own
# paper (TNUJST5101), so this script does not re-run those -- only
# baselines/run_baselines_beauty.sh needs them, since no prior baseline of
# any kind exists for Beauty.
#
# Idempotent: each run is skipped if its summary file already exists, so
# re-running this script (e.g. after a Kaggle session gets cut off) is safe.
# That check relies on outputs/... resolving to the same path every run --
# cd to the repo root first so it does, regardless of which directory the
# caller invoked `bash .../run_baselines_education.sh` from.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Kaggle's default transformers build fails to load Salesforce/codet5-base's
# tokenizer (TypeError: extra_special_tokens must be a list/tuple ...) --
# pin to a version confirmed working against every model used below.
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate

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
    --output_dir "$out" --model_name vinai/phobert-base-v2 --segmenter pyvi --seeds 42

echo "[$(date '+%H:%M:%S')] === BERT-based: XLM-R ==="
out=outputs/bert_education_xlmr
skip_if_exists "$out/multi_seed_summary.json" || \
python3 baselines/bert_baseline.py --mode train --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
    --output_dir "$out" --model_name xlm-roberta-base --segmenter none --seeds 42

echo "[$(date '+%H:%M:%S')] === BERT-based: Ensemble BERTs ==="
out=outputs/bert_education_ensemble
skip_if_exists "$out/test_metrics.json" || \
python3 baselines/bert_baseline.py --mode ensemble --test_path "$TEST" --output_dir "$out" \
    --checkpoints outputs/bert_education_phobert/best_model.pt outputs/bert_education_xlmr/best_model.pt

echo "[$(date '+%H:%M:%S')] === T5-based: mT5-large / viT5-large / viT5-base ==="
run_t5_seq2seq() {
    local name="$1" model="$2"
    local out="outputs/${name}_education"
    skip_if_exists "$out/multi_seed_summary.json" || \
    python3 baselines/t5_seq2seq_baseline.py --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
        --output_dir "$out" --model_name "$model" --seeds 42
}
run_t5_seq2seq mt5large  google/mt5-large
run_t5_seq2seq vit5large VietAI/vit5-large
run_t5_seq2seq vit5base  VietAI/vit5-base

echo "[$(date '+%H:%M:%S')] === Instruction tuning: 4 variants ==="
for format in code nl; do
    for lang in vi en; do
        out="outputs/t5_education_${format}_${lang}"
        skip_if_exists "$out/multi_seed_summary.json" || \
        python3 baselines/t5_instruction_tuning.py --domain "$DOMAIN_NAME" --format "$format" --lang "$lang" \
            --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" --output_dir "$out" --seeds 42
    done
done

echo "[$(date '+%H:%M:%S')] Education baselines finished."
