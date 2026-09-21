#!/usr/bin/env bash
# Kaggle session 2 (optional): design ablations on Restaurant and Hotel, 5 seeds each, fixed fusion, descriptions.
#   gate_hard  sentiment head reads z * 1[p_acd >= 0.5]
#   gate_none  sentiment head reads z (no ACD -> sentiment interaction)
#   query_id   learned free embedding per category instead of the category text
# 3 ablations x 2 domains x 5 seeds = 30 trainings. Compare with outputs/cage_<domain>_fixed (matched reference).
#   bash run_cage_session2.sh          DRY=1 bash run_cage_session2.sh   (print commands only)
# Resumable: relaunch the same command if the session is cut.
set -uo pipefail
cd "$(dirname "$0")"
bash run_cage_ablation.sh all restaurant hotel
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh session2
