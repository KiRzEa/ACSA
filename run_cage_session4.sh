#!/usr/bin/env bash
# Kaggle session 4: adapter-size ablation on Restaurant (25 trainings) + design ablations on Education (15 trainings).
#   bash run_cage_session4.sh      DRY=1 bash run_cage_session4.sh
set -uo pipefail; cd "$(dirname "$0")"
bash run_cage_ablation.sh adapter restaurant
bash run_cage_ablation.sh design education
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh session4
