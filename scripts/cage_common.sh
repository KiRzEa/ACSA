# Shared by run_cage_5seeds_description.sh and run_cage_5seeds_name.sh (source, do not run directly).
# Expects: CATEGORY_TEXT (description|name), VARIANTS (e.g. "fixed learned"), OUT_PREFIX (e.g. cage).
# Optional: EXTRA_ARGS (extra flags for train_mtl_acsa_v2.py, e.g. "--gate_mode hard").
# Protocol (identical for both scripts): PhoBERT-base-v2, 5 seeds, 10 epochs, effective batch 16 = BATCH x ACCUM,
# max length 256, default GradNorm / BCE-ACD / 8-head / 192-d adapter config. Only --category_text differs.
set -uo pipefail

MODEL=vinai/phobert-base-v2
SEEDS="42,123,2024,7,99"
BATCH="${BATCH:-16}"            # BATCH=8 -> ACCUM=2 if GPU memory is tight
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

run_variant() {   # run_variant <domain> <variant: fixed|learned>
  local domain="$1" variant="$2" d out extra=()
  d="$(data_dir "$domain")"
  out="outputs/${OUT_PREFIX}_${domain}_${variant}"
  [ "$variant" = "learned" ] && extra+=(--learned_fusion)
  if [ -f "${out}/multi_seed_summary.json" ]; then
    echo "[$(date '+%H:%M:%S')] SKIP  ${domain}/${variant} (multi_seed_summary.json exists)"; return
  fi
  local cmd=(python3 train_mtl_acsa_v2.py
    --train_path "${d}/Train.txt" --dev_path "${d}/Dev.txt" --test_path "${d}/Test.txt"
    --model_name "$MODEL" --seeds "$SEEDS" --output_dir "$out" --domain "$domain"
    --category_text "$CATEGORY_TEXT"
    --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" --max_length "$MAXLEN" ${EXTRA_ARGS:-} ${extra[@]+"${extra[@]}"})
  if [ "${DRY:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  echo "[$(date '+%H:%M:%S')] START ${domain}/${variant} (category_text=${CATEGORY_TEXT})"
  SECONDS=0
  "${cmd[@]}"
  echo "[$(date '+%H:%M:%S')] DONE  ${domain}/${variant} (${SECONDS}s)"
  # best_model.pt is ~0.5 GB per seed; 50 runs would exceed Kaggle's 20 GB output limit. Drop it once the
  # variant has finished unless KEEP_CKPT=1.
  if [ "${KEEP_CKPT:-0}" != "1" ] && [ -f "${out}/multi_seed_summary.json" ]; then
    find "${out}" -name best_model.pt -delete
  fi
}

DOMAINS=("$@")
[ ${#DOMAINS[@]} -eq 0 ] && DOMAINS=(education restaurant hotel phone beauty)
for domain in "${DOMAINS[@]}"; do
  if [ -z "$(data_dir "$domain")" ]; then echo "unknown domain: $domain"; continue; fi
  for v in $VARIANTS; do run_variant "$domain" "$v"; done
done
echo "[$(date '+%H:%M:%S')] finished: ${DOMAINS[*]} (${CATEGORY_TEXT}, variants: ${VARIANTS})"
