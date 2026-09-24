#!/usr/bin/env bash
# Kaggle (GPU, 2 x T4): baseline seed top-up. One session = one call, each <= ~10.5 h by construction.
#   python3 scripts/kaggle_queue.py plan      (local; prints the session split, currently 12 sessions)
#   bash run_kaggle_baselines.sh <N>          N = 1 .. number of sessions;  BUDGET=10.5 (hours) and DRY=1 are optional
# A job that no longer fits the remaining budget is left for a later session, and the small result
# files are re-packed to /kaggle/working/cage_results_queue_s<N>.zip after every job. Unzip each zip into ./outputs/ locally.
set -uo pipefail
cd "$(dirname "$0")"
N="${1:?usage: bash run_kaggle_baselines.sh <session number>}"
if [ "${DRY:-0}" = "1" ]; then python3 scripts/kaggle_queue.py run --session "$N" --dry-run; exit 0; fi
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate scipy
python3 scripts/kaggle_queue.py run --session "$N" --budget-hours "${BUDGET:-10.5}"
bash scripts/pack_results.sh "queue_s${N}"
