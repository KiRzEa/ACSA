#!/usr/bin/env bash
# Pack only what the paper needs from outputs/cage_* into ONE zip (no checkpoints, no logs of weights):
# metrics.json, multi_seed_summary.json, dev/test_predictions.jsonl, category_texts.json, history.json.
# test_predictions.jsonl is included DELIBERATELY, not just the aggregate metrics -- if an evaluation/parsing bug
# is found later (e.g. a category-string mismatch), the raw predictions let us recompute the correct metric
# without re-running training. NEVER drop it from the -i list below.
# paraphrase_eval.json (scripts/paraphrase_robustness.py) is small like query_swap.json, always included.
# best_model.pt is the one deliberate exception to "no checkpoints": outputs/cage_ckpt_*/ (see
# run_cage_checkpoint.sh) is the sole family of runs that needs its checkpoint pulled off Kaggle at all
# (scripts/paraphrase_robustness.py reloads it locally) -- the pattern below only matches that prefix,
# every other cage_*/ family stays checkpoint-free.
# Run at the end of a Kaggle session:  bash scripts/pack_results.sh [tag]   -> /kaggle/working/cage_results_<tag>.zip
# (falls back to ./cage_results_<tag>.zip when not on Kaggle)
set -euo pipefail
TAG="${1:-$(date +%Y%m%d_%H%M)}"
DEST="/kaggle/working"; [ -d "$DEST" ] || DEST="."
OUT="${DEST}/cage_results_${TAG}.zip"
shopt -s nullglob
zip -q -r "$OUT" outputs/* \
  -i '*_job_times.tsv' '*/test_metrics.json' '*/query_swap.json' '*/paraphrase_eval.json' '*/metrics.json' '*/multi_seed_summary.json' '*/test_predictions.jsonl' '*/dev_predictions.jsonl' '*/category_texts.json' '*/history.json' 'outputs/cage_ckpt_*/seed_*/best_model.pt'
ls -lh "$OUT"; unzip -l "$OUT" | tail -1
