#!/usr/bin/env bash
# Zero-shot leave-one-category-out ablation (see plan: paper/../.claude/plans -- zero-shot generalization
# experiment). For 3 categories per domain (stratified by train-split support: most-frequent / median /
# rarest, restricted to categories with >=15 test occurrences so F1 stays well-defined), trains CAGE
# fixed-fusion with that ONE category's training labels entirely withheld (--holdout_category), then
# evaluates normally on dev/test (which still contain real gold for it). Compare the resulting category-
# restricted F1 against the same category's F1 in the fully-supervised outputs/cage_<domain>_fixed run via
# scripts/zeroshot_stats.py.
#
# Protocol: PhoBERT-base-v2, CATEGORY_TEXT=description, fixed fusion, 10 epochs, max length 256, effective
# batch 16 -- identical to run_cage_5seeds_description.sh except: 3 seeds (not 5; diagnostic sweep, not a
# headline Table 4 number) and --holdout_category set. 5 domains x 3 categories x 3 seeds = 45 runs.
#
#   bash run_cage_zeroshot.sh                  # all domains, all 3 tiers
#   bash run_cage_zeroshot.sh education beauty  # only these domains
#   TIERS="most rarest" bash run_cage_zeroshot.sh  # skip the median tier
#   DRY=1 bash run_cage_zeroshot.sh             # print commands only
# Resumable PER SEED, not just per combo: a combo is only skipped entirely once ALL 3 seeds have a real
# metrics.json + test_predictions.jsonl (checked directly, not just multi_seed_summary.json's existence --
# some combos were partially run on Modal 2026-10-01/02 and left a STALE multi_seed_summary.json behind
# that only reflects whichever single seed ran last, which would otherwise make this script wrongly skip
# the whole combo). Otherwise it still invokes train_mtl_acsa_v2.py with the full 3-seed list --
# train_mtl_acsa_v2.py's own run() already skips any individual seed whose metrics.json/
# test_predictions.jsonl already exist (see its --no_resume flag) and only trains what's missing, then
# rebuilds a correct multi_seed_summary.json from all 3 seeds' metrics.json (fresh or reused) at the end.
set -uo pipefail

MODEL=vinai/phobert-base-v2
SEEDS="42,123,2024"
BATCH="${BATCH:-16}"
ACCUM=$((16 / BATCH))
EPOCHS=10
MAXLEN=256
TIERS="${TIERS:-most median rarest}"

data_dir() {
  case "$1" in
    restaurant) echo Res_ABSA ;;   hotel) echo Hotel_ABSA ;;   phone) echo Phone_ABSA ;;
    education)  echo Education_ABSA ;;   beauty) echo Beauty_ABSA ;;
    *) echo "" ;;
  esac
}

# domain|tier|category|slug -- computed once from Train.txt/Test.txt support counts (train count desc,
# restricted to categories with >=15 test occurrences); see plan file for the selection script.
read -r -d '' TABLE <<'EOF'
restaurant|most|FOOD#QUALITY|food_quality
restaurant|median|RESTAURANT#MISCELLANEOUS|restaurant_miscellaneous
restaurant|rarest|DRINKS#PRICES|drinks_prices
hotel|most|SERVICE#GENERAL|service_general
hotel|median|FOOD&DRINKS#STYLE&OPTIONS|food_drinks_style_options
hotel|rarest|FACILITIES#CLEANLINESS|facilities_cleanliness
phone|most|GENERAL|general
phone|median|PRICE|price
phone|rarest|STORAGE|storage
education|most|Hành vi|hanh_vi
education|median|Nói chung|noi_chung
education|rarest|Cung cấp tài liệu|cung_cap_tai_lieu
beauty|most|Màu sắc|mau_sac
beauty|median|Bao bì|bao_bi
beauty|rarest|Độ bền màu|do_ben_mau
EOF

all_seeds_done() {   # all_seeds_done <out_dir> -- true iff every seed in $SEEDS has metrics.json + test_predictions.jsonl
  local out="$1" s
  IFS=',' read -ra _seeds <<< "$SEEDS"
  for s in "${_seeds[@]}"; do
    [ -f "${out}/seed_${s}/metrics.json" ] && [ -f "${out}/seed_${s}/test_predictions.jsonl" ] || return 1
  done
  return 0
}

run_combo() {   # run_combo <domain> <tier> <category> <slug>
  local domain="$1" tier="$2" category="$3" slug="$4" d out
  d="$(data_dir "$domain")"
  out="outputs/cage_zeroshot_${domain}_${slug}"
  if all_seeds_done "$out"; then
    echo "[$(date '+%H:%M:%S')] SKIP  ${domain}/${tier} (${category}) (all seeds already complete)"; return
  fi
  local cmd=(python3 train_mtl_acsa_v2.py
    --train_path "${d}/Train.txt" --dev_path "${d}/Dev.txt" --test_path "${d}/Test.txt"
    --model_name "$MODEL" --seeds "$SEEDS" --output_dir "$out" --domain "$domain"
    --category_text description --holdout_category "$category"
    --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" --max_length "$MAXLEN")
  if [ "${DRY:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  echo "[$(date '+%H:%M:%S')] START ${domain}/${tier} (holdout=${category})"
  SECONDS=0
  "${cmd[@]}"
  echo "[$(date '+%H:%M:%S')] DONE  ${domain}/${tier} (${SECONDS}s)"
  # Same 0.5GB/seed checkpoint-limit rationale as cage_common.sh -- this experiment doesn't need checkpoints.
  if [ -f "${out}/multi_seed_summary.json" ]; then find "${out}" -name best_model.pt -delete; fi
}

DOMAINS=("$@")
[ ${#DOMAINS[@]} -eq 0 ] && DOMAINS=(education restaurant hotel phone beauty)
while IFS='|' read -r domain tier category slug; do
  [ -z "$domain" ] && continue
  match=0
  for want in "${DOMAINS[@]}"; do [ "$want" = "$domain" ] && match=1; done
  [ "$match" = "0" ] && continue
  tier_match=0
  for t in $TIERS; do [ "$t" = "$tier" ] && tier_match=1; done
  [ "$tier_match" = "0" ] && continue
  if [ -z "$(data_dir "$domain")" ]; then echo "unknown domain: $domain"; continue; fi
  run_combo "$domain" "$tier" "$category" "$slug"
done <<< "$TABLE"
echo "[$(date '+%H:%M:%S')] finished: ${DOMAINS[*]} (tiers: ${TIERS})"
