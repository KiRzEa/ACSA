#!/usr/bin/env bash
# Re-runs the BERT-family ensemble baseline (PhoBERT + XLM-R) for all five
# domains, single seed (42, not best-of-3), purely to obtain per-example
# test_predictions.jsonl for the case study baseline-comparison column --
# the headline Table 3/4 numbers are already final and unaffected by this.
#
# Disk-conscious for a single Kaggle session: each domain's two checkpoints
# are deleted immediately after that domain's ensemble step consumes them,
# so at most 2 checkpoints (not 10) ever sit on disk at once, and the HF
# cache is cleared between domains too.
#
# Idempotent: a domain is skipped if its ensemble test_predictions.jsonl
# already exists, so re-running this script is safe WITHIN the same
# session (a brand new session starts with an empty outputs/, so this
# can't detect runs completed in a previous session).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate

run_domain() {
    local name=$1 data_dir=$2
    local out_ens="outputs/bert_${name}_ensemble"

    if [ -f "$out_ens/test_predictions.jsonl" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP $name (already exists: $out_ens/test_predictions.jsonl)"
        return
    fi

    echo "[$(date '+%H:%M:%S')] $name: training PhoBERT..."
    python3 baselines/bert_baseline.py --mode train \
        --train_path "${data_dir}/Train.txt" --dev_path "${data_dir}/Dev.txt" \
        --test_path "${data_dir}/Test.txt" --output_dir "outputs/bert_${name}_phobert" \
        --model_name vinai/phobert-base-v2 --segmenter pyvi --seeds 42

    echo "[$(date '+%H:%M:%S')] $name: training XLM-R..."
    python3 baselines/bert_baseline.py --mode train \
        --train_path "${data_dir}/Train.txt" --dev_path "${data_dir}/Dev.txt" \
        --test_path "${data_dir}/Test.txt" --output_dir "outputs/bert_${name}_xlmr" \
        --model_name xlm-roberta-base --segmenter none --seeds 42

    echo "[$(date '+%H:%M:%S')] $name: ensembling..."
    python3 baselines/bert_baseline.py --mode ensemble \
        --checkpoints "outputs/bert_${name}_phobert/best_model.pt" "outputs/bert_${name}_xlmr/best_model.pt" \
        --test_path "${data_dir}/Test.txt" --output_dir "$out_ens"

    # test_predictions.jsonl is now in $out_ens/ -- drop the checkpoints
    # before starting the next domain so peak disk stays at ~2 checkpoints.
    rm -f "outputs/bert_${name}_phobert/best_model.pt" "outputs/bert_${name}_xlmr/best_model.pt"
    rm -rf ~/.cache/huggingface/hub

    echo "[$(date '+%H:%M:%S')] DONE $name -> $out_ens/test_predictions.jsonl"
}

run_domain restaurant Res_ABSA
run_domain hotel      Hotel_ABSA
run_domain phone      Phone_ABSA
run_domain education  Education_ABSA
run_domain beauty     Beauty_ABSA

echo "[$(date '+%H:%M:%S')] All domains done."
