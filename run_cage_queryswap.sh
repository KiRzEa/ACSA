#!/usr/bin/env bash
# Counterfactual query substitution (advisor request): CAGE fixed fusion, 5 seeds, protocol of run_cage_5seeds_description.sh,
# plus --query_swap_eval, which after testing re-evaluates every slot with a substituted query (own / nearest / farthest /
# random / mean / zero) and writes <run>/seed_<s>/query_swap.json.  Aggregate with scripts/query_swap_stats.py.
#   bash run_cage_queryswap.sh [domain ...]            description queries -> outputs/cage_qs_<domain>_fixed/
#   QUERY=id bash run_cage_queryswap.sh [domain ...]   free-embedding control -> outputs/cage_qsid_<domain>_fixed/
#   DRY=1 prints the commands.  Resumable.  25 trainings for description (5 domains x 5 seeds).
CATEGORY_TEXT=description
VARIANTS=fixed
if [ "${QUERY:-text}" = "id" ]; then OUT_PREFIX=cage_qsid; EXTRA_ARGS="--query_swap_eval --category_query id"
else OUT_PREFIX=cage_qs; EXTRA_ARGS="--query_swap_eval"; fi
source "$(dirname "$0")/scripts/cage_common.sh"
