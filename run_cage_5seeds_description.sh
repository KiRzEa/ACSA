#!/usr/bin/env bash
# CAGE, 5 seeds, category DESCRIPTION text (mapper.CATEGORY_DESCRIPTIONS). Main results of the paper.
# Default: fixed AND learned fusion, 5 domains = 50 runs -> outputs/cage_<domain>_<fixed|learned>/
#   bash run_cage_5seeds_description.sh                # all domains
#   bash run_cage_5seeds_description.sh education      # one or more domains
#   VARIANTS=fixed bash run_cage_5seeds_description.sh # fixed fusion only
#   DRY=1 bash run_cage_5seeds_description.sh          # print commands only
# Resumable (a variant with multi_seed_summary.json is skipped). Suggested Kaggle split (12 h sessions):
#   A: education restaurant   B: hotel phone   C: beauty
CATEGORY_TEXT=description
VARIANTS="${VARIANTS:-fixed learned}"
OUT_PREFIX=cage
source "$(dirname "$0")/scripts/cage_common.sh"
