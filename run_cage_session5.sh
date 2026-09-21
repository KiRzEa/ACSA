#!/usr/bin/env bash
# Kaggle session 5 (optional, heavy): design ablations on Phone and Beauty (30 trainings; long document-level reviews).
#   bash run_cage_session5.sh      DRY=1 bash run_cage_session5.sh
set -uo pipefail; cd "$(dirname "$0")"
bash run_cage_ablation.sh design phone beauty
[ "${DRY:-0}" = "1" ] || bash scripts/pack_results.sh session5
