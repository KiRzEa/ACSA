#!/usr/bin/env bash
# Pack only what the paper needs from outputs/cage_* into ONE zip (no checkpoints, no logs of weights):
# metrics.json, multi_seed_summary.json, dev/test_predictions.jsonl, category_texts.json, history.json.
# Run at the end of a Kaggle session:  bash scripts/pack_results.sh [tag]   -> /kaggle/working/cage_results_<tag>.zip
# (falls back to ./cage_results_<tag>.zip when not on Kaggle)
set -euo pipefail
TAG="${1:-$(date +%Y%m%d_%H%M)}"
DEST="/kaggle/working"; [ -d "$DEST" ] || DEST="."
OUT="${DEST}/cage_results_${TAG}.zip"
shopt -s nullglob
zip -q -r "$OUT" outputs/* \
  -i '*_job_times.tsv' '*/test_metrics.json' '*/query_swap.json' '*/metrics.json' '*/multi_seed_summary.json' '*/test_predictions.jsonl' '*/dev_predictions.jsonl' '*/category_texts.json' '*/history.json'
ls -lh "$OUT"; unzip -l "$OUT" | tail -1
