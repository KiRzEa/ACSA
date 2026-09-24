#!/usr/bin/env bash
# Kaggle: CAGE (fixed fusion, description queries, 5 seeds) with --query_swap_eval; writes query_swap.json per seed.
#   bash run_kaggle_queryswap.sh A        restaurant hotel education   (15 trainings)
#   bash run_kaggle_queryswap.sh B        phone beauty                 (10 trainings)
#   bash run_kaggle_queryswap.sh A id     same, free-embedding control (--category_query id), outputs/cage_qsid_*
# One call per session. DRY=1 prints the commands.
# Aggregate locally: python3 scripts/query_swap_stats.py [--prefix cage_qsid]
set -uo pipefail
cd "$(dirname "$0")"
PART="${1:?usage: bash run_kaggle_queryswap.sh <A|B> [id]}"
case "$PART" in A) DOMAINS="restaurant hotel education" ;; B) DOMAINS="phone beauty" ;; *) echo "A or B"; exit 1 ;; esac
[ "${2:-}" = "id" ] && export QUERY=id
[ "${DRY:-0}" = "1" ] || pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate scipy
bash run_cage_queryswap.sh $DOMAINS
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh "queryswap_${PART}${2:-}"
