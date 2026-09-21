#!/usr/bin/env bash
# Experiments still missing on 2026-09-21 (see paper/PAPER_TODO.md). Same protocol as run_cage_5seeds_description.sh.
#   phone     Phone, description, learned fusion: seeds 2024, 7, 99 (seeds 42 and 123 are reused if their folders
#             outputs/cage_phone_learned/seed_42 and seed_123 are present in this session; otherwise all 5 seeds are trained)
#   beauty    Beauty, description, fixed + learned fusion, 5 seeds each (10 trainings; the largest domain, give it a session)
#   ablation  design ablations gate_hard, gate_none, query_id on Restaurant and Hotel, 5 seeds each (30 trainings; optional)
# Usage:  bash run_cage_missing.sh phone
#         bash run_cage_missing.sh beauty
#         bash run_cage_missing.sh ablation
#         bash run_cage_missing.sh all        # phone, then beauty, then ablation (needs more than one 12 h session)
#         DRY=1 bash run_cage_missing.sh all  # print the commands only
# Every step is resumable: finished variants (multi_seed_summary.json) and finished seeds (metrics.json) are skipped.
# At the end a small zip (metrics + predictions, no checkpoints) is written by scripts/pack_results.sh.
set -uo pipefail
cd "$(dirname "$0")"
STEP="${1:-all}"

run_phone()    { VARIANTS=learned bash run_cage_5seeds_description.sh phone; }
run_beauty()   { bash run_cage_5seeds_description.sh beauty; }
run_ablation() { bash run_cage_ablation.sh all restaurant hotel; }

case "$STEP" in
  phone)    run_phone ;;
  beauty)   run_beauty ;;
  ablation) run_ablation ;;
  all)      run_phone; run_beauty; run_ablation ;;
  *) echo "usage: bash run_cage_missing.sh {phone|beauty|ablation|all}"; exit 1 ;;
esac

if [ "${DRY:-0}" != "1" ]; then bash scripts/pack_results.sh "missing_${STEP}"; fi
