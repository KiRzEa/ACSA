#!/usr/bin/env bash
# CAGE, 5 seeds, category NAME text (raw category name, no description). Ablation: name vs description.
# Default: fixed fusion only, 5 domains = 25 runs -> outputs/cage_<domain>_name_fixed/
# (the matched description runs are outputs/cage_<domain>_fixed/ from run_cage_5seeds_description.sh)
#   bash run_cage_5seeds_name.sh restaurant              # e.g. only Restaurant
#   VARIANTS="fixed learned" bash run_cage_5seeds_name.sh  # also learned fusion (adds 25 runs)
#   DRY=1 bash run_cage_5seeds_name.sh
CATEGORY_TEXT=name
VARIANTS="${VARIANTS:-fixed}"
OUT_PREFIX=cage_name
source "$(dirname "$0")/scripts/cage_common.sh"
