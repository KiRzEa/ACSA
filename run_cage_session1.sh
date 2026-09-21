#!/usr/bin/env bash
# Kaggle session 1: Phone learned fusion (seeds 2024, 7, 99) + Beauty with descriptions (fixed and learned).
# Needs the finished seeds of Phone learned (outputs/cage_phone_learned/seed_42 and seed_123) to be present first,
# otherwise Phone learned trains all 5 seeds again (see the unzip step in the commands).
#   bash run_cage_session1.sh          DRY=1 bash run_cage_session1.sh   (print commands only)
# Resumable: relaunch the same command if the session is cut.
set -uo pipefail
cd "$(dirname "$0")"
VARIANTS=learned bash run_cage_5seeds_description.sh phone      # 3 trainings (seeds 2024, 7, 99)
bash run_cage_5seeds_description.sh beauty                      # 10 trainings (fixed, then learned)
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh session1
