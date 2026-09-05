#!/usr/bin/env bash
# Education baselines for Table 4 -- remainder only. BERT-based (PhoBERT/
# XLM-R/Ensemble BERTs) is done and already recorded in the paper (PhoBERT:
# 86.32/81.19/83.67, XLM-R: 82.26/80.36/81.30, Ensemble BERTs:
# 87.05/83.17/85.06); Statistics rows (SVM/CNN/BiLSTM-CNN) have real
# published numbers from the dataset's own paper (TNUJST5101). What's left:
# viT5-base and all 4 instruction-tuning variants (none of Education's have
# run yet). mT5-large and viT5-large each run in their own dedicated script
# now (run_baselines_education_mt5large.sh /
# run_baselines_education_vit5large.sh) -- run those as separate Kaggle
# sessions, not alongside this one or each other, so a crash in one doesn't
# cost progress on the other.
#
# viT5-base and the 4 instruction-tuning variants (using codet5-base/
# viT5-base, both small enough for a single GPU) run two at a time, one
# pinned to each GPU via CUDA_VISIBLE_DEVICES=0/1.
#
# Idempotent: skipped if its summary file already exists, so re-running this
# script is safe WITHIN the same session (a brand new session starts with an
# empty outputs/, so this can't detect runs completed in a previous session
# -- tell me the results instead and I'll record them / trim the script).
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

# Called only at top-level, after a `wait` (i.e. nothing still running) --
# each experiment re-downloads whatever checkpoint it needs, uses it, writes
# its results, then this clears the HF cache before the next group starts,
# trading a bit of re-download bandwidth for guaranteed disk headroom --
# leaving caches around across a full session is what exhausted Kaggle's
# disk ("No space left on device") in an earlier run. NEVER call this from
# inside a backgrounded run_* job -- two parallel jobs share one cache dir,
# and clearing it while the other is still mid-download would delete its
# in-progress files out from under it.
clear_hf_cache() {
    rm -rf ~/.cache/huggingface/hub 2>/dev/null || true
}

run_t5_seq2seq() {
    local name="$1" model="$2"; shift 2
    local out="outputs/${name}_education"
    skip_if_exists "$out/multi_seed_summary.json" && return 0
    echo "[$(date '+%H:%M:%S')] START t5_seq2seq $name (GPU ${CUDA_VISIBLE_DEVICES:-all})"
    python3 baselines/t5_seq2seq_baseline.py --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" \
        --output_dir "$out" --model_name "$model" --seeds 42 "$@"
    echo "[$(date '+%H:%M:%S')] DONE  t5_seq2seq $name"
}

run_instruction() {
    local format="$1" lang="$2"
    local out="outputs/t5_education_${format}_${lang}"
    skip_if_exists "$out/multi_seed_summary.json" && return 0
    echo "[$(date '+%H:%M:%S')] START instruction $format/$lang (GPU ${CUDA_VISIBLE_DEVICES:-all})"
    python3 baselines/t5_instruction_tuning.py --domain "$DOMAIN_NAME" --format "$format" --lang "$lang" \
        --train_path "$TRAIN" --dev_path "$DEV" --test_path "$TEST" --output_dir "$out" --seeds 42
    echo "[$(date '+%H:%M:%S')] DONE  instruction $format/$lang"
}

echo "[$(date '+%H:%M:%S')] === T5-based: viT5-base + Instruction: group 1/3 (Code-Vi), parallel ==="
CUDA_VISIBLE_DEVICES=0 run_t5_seq2seq vit5base VietAI/vit5-base &
CUDA_VISIBLE_DEVICES=1 run_instruction code vi &
wait
clear_hf_cache

echo "[$(date '+%H:%M:%S')] === Instruction tuning: group 2/3 (Code-En + NL-Vi, parallel) ==="
CUDA_VISIBLE_DEVICES=0 run_instruction code en &
CUDA_VISIBLE_DEVICES=1 run_instruction nl vi &
wait
clear_hf_cache

echo "[$(date '+%H:%M:%S')] === Instruction tuning: group 3/3 (NL-En) ==="
CUDA_VISIBLE_DEVICES=0 run_instruction nl en

echo "[$(date '+%H:%M:%S')] Education baselines finished."
