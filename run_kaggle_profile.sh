#!/usr/bin/env bash
# Kaggle (GPU, 2 x T4): training cost and inference latency of CAGE and every baseline trained in this work, measured
# under one protocol on Hotel (largest taxonomy, 34 categories). Each model is trained for ONE epoch with the exact
# configuration of its five-seed runs, then timed on 200 test reviews (see cost_profile.py for the definitions).
# Models run one after another, never in parallel, so timings do not interfere. Single-GPU models use GPU 0; mT5-large
# and ViT5-large are split over both GPUs as in their real runs. Expected total: about 1.5-2.5 h.
#   bash run_kaggle_profile.sh          -> outputs/profile/*.json, packed into /kaggle/working/cage_profile.zip
set -uo pipefail
cd "$(dirname "$0")"
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate scipy
D="--train_path Hotel_ABSA/Train.txt --dev_path Hotel_ABSA/Dev.txt --test_path Hotel_ABSA/Test.txt"
P=outputs/profile
mkdir -p "$P"
DEST=/kaggle/working; [ -d "$DEST" ] || DEST=.

run() {  # run <name> <gpus: 0 | 0,1> <command...>
  local name="$1" gpus="$2"; shift 2
  if [ -f "$P/$name.json" ]; then echo "[$(date +%T)] SKIP $name (done)"; return; fi
  echo "[$(date +%T)] START $name"
  SECONDS=0
  CUDA_VISIBLE_DEVICES="$gpus" "$@" --profile_json "$P/$name.json" > "$P/$name.log" 2>&1 \
    && echo "[$(date +%T)] DONE  $name (${SECONDS}s)" || { echo "[$(date +%T)] FAIL  $name -- tail of $P/$name.log:"; tail -5 "$P/$name.log"; }
  zip -q -r "$DEST/cage_profile.zip" "$P" -i '*.json' '*.log'   # re-packed after every model, so a cut session keeps what finished
}

CAGE="python3 train_mtl_acsa_v2.py $D --model_name vinai/phobert-base-v2 --seed 42 --domain hotel --category_text description --epochs 10 --batch_size 16 --grad_accum_steps 1 --max_length 256"
run cage_fixed    0 $CAGE --output_dir $P/run_cage_fixed
run cage_learned  0 $CAGE --output_dir $P/run_cage_learned --learned_fusion
run cage_minimal  0 $CAGE --output_dir $P/run_cage_minimal --heads joint_only
run phobert       0 python3 baselines/bert_baseline.py --mode train $D --seeds 42 --output_dir $P/run_phobert --model_name vinai/phobert-base-v2 --segmenter pyvi
run xlmr          0 python3 baselines/bert_baseline.py --mode train $D --seeds 42 --output_dir $P/run_xlmr --model_name xlm-roberta-base --segmenter none
run cnn           0 python3 baselines/cnn_baseline.py $D --seeds 42 --output_dir $P/run_cnn
run bilstm_cnn    0 python3 baselines/cnn_baseline.py $D --seeds 42 --output_dir $P/run_bilstm_cnn --use_bilstm
run vit5base      0 python3 baselines/t5_seq2seq_baseline.py $D --output_dir $P/run_vit5base --model_name VietAI/vit5-base --seeds 42
run instr_nl_vi   0 python3 baselines/t5_instruction_tuning.py --domain 'hotel' --format nl --lang vi $D --output_dir $P/run_instr_nl_vi --seeds 42
run instr_code_vi 0 python3 baselines/t5_instruction_tuning.py --domain 'hotel' --format code --lang vi $D --output_dir $P/run_instr_code_vi --seeds 42
LARGE="--seeds 42 --batch_size 8 --eval_batch_size 4 --epochs 10 --gradient_checkpointing --device_map_auto"
run vit5large   0,1 python3 baselines/t5_seq2seq_baseline.py $D --output_dir $P/run_vit5large --model_name VietAI/vit5-large $LARGE
rm -rf ~/.cache/huggingface/hub/models--VietAI--vit5-large
run mt5large    0,1 python3 baselines/t5_seq2seq_baseline.py $D --output_dir $P/run_mt5large --model_name google/mt5-large $LARGE

echo "[$(date +%T)] finished; results:"; ls -la "$P"/*.json; ls -la "$DEST/cage_profile.zip"
