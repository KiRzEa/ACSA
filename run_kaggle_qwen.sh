#!/usr/bin/env bash
# Kaggle (GPU T4 x2, Internet on): Qwen2.5-3B-Instruct + LoRA (Unsloth) instruction-tuned baselines.
# Two jobs run in parallel, one per T4 (Unsloth trains on a single GPU per process).
#   local:  python3 scripts/kaggle_queue.py plan --groups qwen --round q1                 (first round, estimates)
#           python3 scripts/kaggle_queue.py plan --groups qwen --round q2 --calibrate outputs/_job_times.tsv
#   Kaggle: bash run_kaggle_qwen.sh <N>        N = session number; BUDGET=10.5 (hours) and DRY=1 are optional
# Results are re-packed to /kaggle/working/cage_results_queue_<round>_s<N>.zip after every job; unzip into ./ locally.
set -uo pipefail
cd "$(dirname "$0")"
N="${1:?usage: bash run_kaggle_qwen.sh <session number>}"
if [ "${DRY:-0}" = "1" ]; then python3 scripts/kaggle_queue.py run --session "$N" --dry-run; exit 0; fi
pip install -q --upgrade unsloth unsloth_zoo
python3 -c "import unsloth, torch; print('unsloth', unsloth.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())" \
  || { echo "Unsloth does not import; fall back with --backend hf --batch_size 8 --grad_accum 2 (see baselines/qwen_instruction_tuning.py)"; exit 1; }
# Download the 4-bit weights once, before the two parallel jobs start, so they do not race on the cache.
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen2.5-3B-Instruct-bnb-4bit')"
python3 scripts/kaggle_queue.py run --session "$N" --budget-hours "${BUDGET:-10.5}"
bash scripts/pack_results.sh "$(python3 scripts/kaggle_queue.py run --session "$N" --tag)"
