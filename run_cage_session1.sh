#!/usr/bin/env bash
# Kaggle session 1: Phone learned fusion (all 5 seeds, the earlier partial run was deleted) + Beauty with descriptions
# (fixed and learned). 5 + 10 = 15 trainings.
#   bash run_cage_session1.sh          DRY=1 bash run_cage_session1.sh   (print commands only)
# Resumable: relaunch the same command if the session is cut.
set -uo pipefail
cd "$(dirname "$0")"
VARIANTS=learned bash run_cage_5seeds_description.sh phone      # 5 trainings (seeds 42, 123, 2024, 7, 99)
bash run_cage_5seeds_description.sh beauty                      # 10 trainings (fixed, then learned)
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh session1
