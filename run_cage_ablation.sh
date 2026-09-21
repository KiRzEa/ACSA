#!/usr/bin/env bash
# CAGE design ablations, 5 seeds each, fixed fusion, description text, otherwise the same protocol as
# run_cage_5seeds_description.sh (whose "fixed" runs are the matched reference).
#   gate_hard   sentiment head reads z * 1[p_acd >= 0.5]        (hard gate)
#   gate_none   sentiment head reads z                          (no ACD -> sentiment interaction)
#   query_id    category query = learned free embedding per category; no category text at all
# Usage: bash run_cage_ablation.sh <gate_hard|gate_none|query_id|all> [domain ...]
#   default domains: restaurant hotel (dense vs sparse taxonomy). Each ablation x domain = 5 training runs.
#   DRY=1 prints commands. Outputs: outputs/cage_abl_<ablation>_<domain>_fixed/
ABL="${1:-all}"; shift || true
DOMS=("$@"); [ ${#DOMS[@]} -eq 0 ] && DOMS=(restaurant hotel)
CATEGORY_TEXT=description
VARIANTS=fixed
list=("$ABL"); [ "$ABL" = "all" ] && list=(gate_hard gate_none query_id)
for a in "${list[@]}"; do
  case "$a" in
    gate_hard) export EXTRA_ARGS="--gate_mode hard" ;;
    gate_none) export EXTRA_ARGS="--gate_mode none" ;;
    query_id)  export EXTRA_ARGS="--category_query id" ;;
    *) echo "unknown ablation: $a"; exit 1 ;;
  esac
  export OUT_PREFIX="cage_abl_${a}"
  export CATEGORY_TEXT VARIANTS
  bash -c 'source "$0/scripts/cage_common.sh"' "$(cd "$(dirname "$0")" && pwd)" "${DOMS[@]}"
done
