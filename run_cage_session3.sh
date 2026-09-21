#!/usr/bin/env bash
# Kaggle session 3: re-run the old Restaurant ablations with 5 seeds.
#   components (lw_fixed, acd_focal, acd_asl, child_tuning) = 20 trainings, heads (2,4,12,16,24) = 25 trainings.
#   bash run_cage_session3.sh      DRY=1 bash run_cage_session3.sh
set -uo pipefail; cd "$(dirname "$0")"
bash run_cage_ablation.sh components restaurant
bash run_cage_ablation.sh heads restaurant
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh session3
