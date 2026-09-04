#!/usr/bin/env bash
# Beauty baselines for Table 4 -- Kaggle session (GPU, T4x2). Fills every row
# of Table 4 that is currently "--" for Beauty: CNN, BiLSTM-CNN, BERT-based
# (PhoBERT / XLM-R / Ensemble BERTs), T5-based (mT5-large / viT5-large /
# viT5-base), and the 4 instruction-tuning variants. The SVM row is CPU-only
# and fast enough to run locally -- see baselines/svm_tfidf_baseline.py, no
# Kaggle session needed for it.
#
# Two ways this uses both GPUs: (1) models small enough for one GPU (CNN,
# BiLSTM-CNN, PhoBERT, XLM-R, viT5-base, codet5-base) run two independent
# jobs at once, one pinned to each GPU via CUDA_VISIBLE_DEVICES=0/1;
# (2) mT5-large/viT5-large are too big for a single ~15GB T4 even alone, so
# those run with --device_map_auto, which splits ONE model's layers across
# BOTH GPUs (real model parallelism, not two separate jobs) -- see the
# LARGE_MEM_ARGS comment below. Ensemble BERTs depends on both PhoBERT and
# XLM-R finishing first, so it runs alone afterward (but is quick --
# inference only, no training).
#
# Idempotent: each run is skipped if its summary file already exists, so
# re-running this script (e.g. after a Kaggle session gets cut off) is safe
# WITHIN the same session (a brand new session starts with an empty
# outputs/, so this can't detect runs completed in a previous session --
# tell me the results instead and I'll record them / comment the step out).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

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

run_cnn() {
    local name="$1"; shift
    local out="outputs/${name}_beauty"
    skip_if_exists "$out/multi_seed_summary.json" && return 0
    echo "[$(date '+%H:%M:%S')] START $name (GPU $CUDA_VISIBLE_DEVICES)"
    python3 baselines/cnn_baseline.py --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
        --output_dir "$out" --seeds 42 "$@"
    echo "[$(date '+%H:%M:%S')] DONE  $name"
}

run_bert() {
    local name="$1" model="$2" segmenter="$3"
    local out="outputs/bert_beauty_${name}"
    skip_if_exists "$out/multi_seed_summary.json" && return 0
    echo "[$(date '+%H:%M:%S')] START bert $name (GPU $CUDA_VISIBLE_DEVICES)"
    python3 baselines/bert_baseline.py --mode train --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
        --output_dir "$out" --model_name "$model" --segmenter "$segmenter" --seeds 42
    echo "[$(date '+%H:%M:%S')] DONE  bert $name"
}

run_t5_seq2seq() {
    local name="$1" model="$2"; shift 2
    local out="outputs/${name}_beauty"
    skip_if_exists "$out/multi_seed_summary.json" && return 0
    echo "[$(date '+%H:%M:%S')] START t5_seq2seq $name (GPU $CUDA_VISIBLE_DEVICES)"
    python3 baselines/t5_seq2seq_baseline.py --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
        --output_dir "$out" --model_name "$model" --seeds 42 "$@"
    echo "[$(date '+%H:%M:%S')] DONE  t5_seq2seq $name"
}

run_instruction() {
    local format="$1" lang="$2"
    local out="outputs/t5_beauty_${format}_${lang}"
    skip_if_exists "$out/multi_seed_summary.json" && return 0
    echo "[$(date '+%H:%M:%S')] START instruction $format/$lang (GPU $CUDA_VISIBLE_DEVICES)"
    python3 baselines/t5_instruction_tuning.py --domain "$DOMAIN_NAME" --format "$format" --lang "$lang" \
        --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" --output_dir "$out" --seeds 42
    echo "[$(date '+%H:%M:%S')] DONE  instruction $format/$lang"
}

# -large checkpoints (~800M-1.2B params) don't fit on a single ~15GB Kaggle
# GPU at all -- AdamW's optimizer state for a ~1.2B-parameter model alone
# already exceeds 15GB before any activations, so batch_size=1 doesn't save
# it. --device_map_auto splits the model's layers across BOTH GPUs (real
# model parallelism, not just "one job per GPU"), so these two must run
# sequentially with neither GPU pinned via CUDA_VISIBLE_DEVICES -- each needs
# both. gradient_checkpointing further trims activation memory on top of that.
LARGE_MEM_ARGS=(--batch_size 2 --eval_batch_size 4 --gradient_checkpointing --device_map_auto)

echo "[$(date '+%H:%M:%S')] === Statistics-based: CNN + BiLSTM-CNN, parallel ==="
CUDA_VISIBLE_DEVICES=0 run_cnn cnn &
CUDA_VISIBLE_DEVICES=1 run_cnn bilstm_cnn --use_bilstm &
wait

echo "[$(date '+%H:%M:%S')] === BERT-based: PhoBERT + XLM-R, parallel ==="
CUDA_VISIBLE_DEVICES=0 run_bert phobert vinai/phobert-base-v2 pyvi &
CUDA_VISIBLE_DEVICES=1 run_bert xlmr    xlm-roberta-base      none &
wait

echo "[$(date '+%H:%M:%S')] === BERT-based: Ensemble BERTs (depends on both above) ==="
out=outputs/bert_beauty_ensemble
skip_if_exists "$out/test_metrics.json" || \
python3 baselines/bert_baseline.py --mode ensemble --test_path "$TEST" --output_dir "$out" \
    --checkpoints outputs/bert_beauty_phobert/best_model.pt outputs/bert_beauty_xlmr/best_model.pt

echo "[$(date '+%H:%M:%S')] === T5-based: mT5-large (both GPUs, model-parallel) ==="
run_t5_seq2seq mt5large  google/mt5-large  "${LARGE_MEM_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] === T5-based: viT5-large (both GPUs, model-parallel) ==="
run_t5_seq2seq vit5large VietAI/vit5-large "${LARGE_MEM_ARGS[@]}"

echo "[$(date '+%H:%M:%S')] === T5-based: viT5-base + Instruction: group 1/3 (Code-Vi), parallel ==="
CUDA_VISIBLE_DEVICES=0 run_t5_seq2seq vit5base VietAI/vit5-base &
CUDA_VISIBLE_DEVICES=1 run_instruction code vi &
wait

echo "[$(date '+%H:%M:%S')] === Instruction tuning: group 2/3 (Code-En + NL-Vi, parallel) ==="
CUDA_VISIBLE_DEVICES=0 run_instruction code en &
CUDA_VISIBLE_DEVICES=1 run_instruction nl vi &
wait

echo "[$(date '+%H:%M:%S')] === Instruction tuning: group 3/3 (NL-En) ==="
CUDA_VISIBLE_DEVICES=0 run_instruction nl en

echo "[$(date '+%H:%M:%S')] Beauty baselines finished."
