#!/usr/bin/env bash
# CAGE: re-run from scratch, 5 seeds x {fixed fusion, learned fusion} x 5 domains
# (50 training runs). Every run uses the default configuration of
# train_mtl_acsa_v2.py, which is the configuration described in the paper's
# training details (GradNorm, BCE ACD loss, 8 cross-attention heads, 192-d
# adapters, no Child-Tuning, no extra vocab). The only difference between the
# two variants is --learned_fusion.
#
# Usage:
#   bash run_cage_5seeds.sh                 # all 5 domains (split across Kaggle sessions, see below)
#   bash run_cage_5seeds.sh education       # one domain
#   bash run_cage_5seeds.sh education beauty
#   DRY=1 bash run_cage_5seeds.sh           # print commands only
#
# Resumable: a variant whose multi_seed_summary.json already exists is skipped,
# so if a Kaggle session is killed just relaunch the same command.
#
# Suggested split for 12h Kaggle sessions (largest last; sizes = training reviews):
#   session A: education (4,002) restaurant (7,035)
#   session B: hotel (7,180)     phone (7,672)
#   session C: beauty (12,787)
# If a session ends early, relaunch it: finished variants are skipped.
#
# Outputs: outputs/cage_<domain>_<variant>/seed_<N>/ (incl. test_predictions.jsonl)
#          outputs/cage_<domain>_<variant>/multi_seed_summary.json
set -uo pipefail  # no -e: one failed run must not stop the rest

MODEL=vinai/phobert-base-v2
SEEDS="42,123,2024,7,99"   # first three match the earlier 3-seed runs

data_dir() {
  case "$1" in
    restaurant) echo Res_ABSA ;;
    hotel)      echo Hotel_ABSA ;;
    phone)      echo Phone_ABSA ;;
    education)  echo Education_ABSA ;;
    beauty)     echo Beauty_ABSA ;;
    *)          echo "" ;;
  esac
}

DOMAINS=("$@")
if [ ${#DOMAINS[@]} -eq 0 ]; then
  DOMAINS=(education restaurant hotel phone beauty)
fi

run() {
  local domain="$1" variant="$2"; shift 2
  local d; d="$(data_dir "$domain")"
  local out="outputs/cage_${domain}_${variant}"
  if [ -f "${out}/multi_seed_summary.json" ]; then
    echo "[$(date '+%H:%M:%S')] SKIP  ${domain}/${variant} (multi_seed_summary.json exists)"
    return
  fi
  local cmd=(python3 train_mtl_acsa_v2.py
    --train_path "${d}/Train.txt" --dev_path "${d}/Dev.txt" --test_path "${d}/Test.txt"
    --model_name "$MODEL" --seeds "$SEEDS" --output_dir "$out" "$@")
  if [ "${DRY:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo; return
  fi
  echo "[$(date '+%H:%M:%S')] START ${domain}/${variant}"
  SECONDS=0
  "${cmd[@]}"
  echo "[$(date '+%H:%M:%S')] DONE  ${domain}/${variant} (${SECONDS}s)"
}

for domain in "${DOMAINS[@]}"; do
  if [ -z "$(data_dir "$domain")" ]; then echo "unknown domain: $domain"; continue; fi
  run "$domain" fixed
  run "$domain" learned --learned_fusion
done

echo "[$(date '+%H:%M:%S')] finished: ${DOMAINS[*]}"
