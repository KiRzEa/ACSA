#!/usr/bin/env bash
# Query-paraphrase-robustness experiment (see plan file, experiment (b)): trains CAGE fixed fusion,
# seed 42 only (matches the already-reported Table 4 config so paraphrase-eval F1 is directly
# comparable), invoking train_mtl_acsa_v2.py DIRECTLY -- not via run_cage_5seeds_description.sh /
# scripts/cage_common.sh, which deletes best_model.pt after every run. This script never deletes it:
# scripts/paraphrase_robustness.py needs it to reload the trained model and re-run inference with
# reworded category queries (see paraphrases.py), which requires NO retraining once the checkpoint
# exists. One checkpoint per domain is enough -- the paraphrase experiment's statistical unit is
# *category* (paired within one model), not *seed*, matching how scripts/query_swap_stats.py already
# treats per-category F1s within a single model as the sample population.
#
#   bash run_cage_checkpoint.sh                 # all domains
#   bash run_cage_checkpoint.sh beauty education # only these domains
#   DRY=1 bash run_cage_checkpoint.sh            # print commands only
# Resumable (a domain with an existing best_model.pt is skipped).
set -uo pipefail

MODEL=vinai/phobert-base-v2
SEED=42
BATCH="${BATCH:-16}"
ACCUM=$((16 / BATCH))
EPOCHS=10
MAXLEN=256

data_dir() {
  case "$1" in
    restaurant) echo Res_ABSA ;;   hotel) echo Hotel_ABSA ;;   phone) echo Phone_ABSA ;;
    education)  echo Education_ABSA ;;   beauty) echo Beauty_ABSA ;;
    *) echo "" ;;
  esac
}

run_domain() {   # run_domain <domain>
  local domain="$1" d base_out out
  d="$(data_dir "$domain")"
  base_out="outputs/cage_ckpt_${domain}_fixed"
  out="${base_out}/seed_${SEED}"   # --seeds (even a single value) makes run() create this seed_<s>/ subdir
  if [ -f "${out}/best_model.pt" ]; then
    echo "[$(date '+%H:%M:%S')] SKIP  ${domain} (best_model.pt exists)"; return
  fi
  local cmd=(python3 train_mtl_acsa_v2.py
    --train_path "${d}/Train.txt" --dev_path "${d}/Dev.txt" --test_path "${d}/Test.txt"
    --model_name "$MODEL" --seeds "$SEED" --output_dir "$base_out" --domain "$domain"
    --category_text description
    --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" --max_length "$MAXLEN")
  if [ "${DRY:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  echo "[$(date '+%H:%M:%S')] START ${domain}"
  SECONDS=0
  "${cmd[@]}"
  echo "[$(date '+%H:%M:%S')] DONE  ${domain} (${SECONDS}s)"
  if [ ! -f "${out}/best_model.pt" ]; then
    echo "WARNING: ${out}/best_model.pt not found after training -- check train_mtl_acsa_v2.py's save_checkpoint() ran"
  fi
}

DOMAINS=("$@")
[ ${#DOMAINS[@]} -eq 0 ] && DOMAINS=(education restaurant hotel phone beauty)
for domain in "${DOMAINS[@]}"; do
  if [ -z "$(data_dir "$domain")" ]; then echo "unknown domain: $domain"; continue; fi
  run_domain "$domain"
done
echo "[$(date '+%H:%M:%S')] finished: ${DOMAINS[*]}"
