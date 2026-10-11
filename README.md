# CAGE: Category-conditioned Attention with Gated Estimation

Code and data for reproducing the experiments of the paper *CAGE: Category-Conditioned Attention with Gated
Estimation for Vietnamese Aspect Category Sentiment Analysis*. CAGE encodes each review once with PhoBERT-base-v2, builds one query
per category from its natural-language description with the same encoder, and lets each query attend over the
review's token states; three task heads (category detection, sentiment, joint) are fused into the final
(category, polarity) predictions.

## Setup

```bash
pip install -r requirements.txt
```

All experiments were run on Kaggle (2 x NVIDIA T4, 16 GB each). One T4 is enough for every model except mT5-large and
ViT5-large, which are split over both GPUs (`--device_map_auto`).

## Data

Five Vietnamese ACSA datasets, already split into `Train.txt`, `Dev.txt` and `Test.txt`:

| Folder | Domain | Source |
|---|---|---|
| `Res_ABSA/` | Restaurant | Thin et al., ACM TALLIP 20(4), 2021 |
| `Hotel_ABSA/` | Hotel | Thin et al., ACM TALLIP 20(4), 2021 |
| `Phone_ABSA/` | Phone | Phan et al., KSEM 2021 (SA2SL) |
| `Education_ABSA/` | Education | Sau et al., TNU J. Sci. Technol. 226(18), 2021 |
| `Beauty_ABSA/` | Beauty | Tran et al., PACLIC 36, 2022 |

Each review is a block of three lines (`#id`, text, labels such as `{ROOMS#CLEANLINESS, negative}`), blocks separated
by a blank line. Full references are in `Read_me.txt`. The category descriptions used as CAGE's queries are in
`mapper.py`.

## Reproducing the results

Every run writes `outputs/<run>/seed_<s>/` with `metrics.json` and `test_predictions.jsonl`; all reported scores are
micro-F1 over sets of (category, polarity) pairs computed from `test_predictions.jsonl`. Five seeds are used
throughout (42, 123, 2024, 7, 99). Training on GPU is not bit-for-bit deterministic, so re-runs agree with the paper
up to seed-level noise.

**CAGE (Table of main results, query substitution).** Fixed fusion with query substitution on the same models, then
learned fusion:

```bash
OUT_PREFIX=cage_v2 EXTRA_ARGS=--query_swap_eval VARIANTS=fixed bash run_cage_5seeds_description.sh
VARIANTS=learned bash run_cage_5seeds_description.sh
bash run_cage_5seeds_name.sh                     # category name instead of description
```

**Ablations.** Each group trains five seeds per configuration (the header of `run_cage_ablation.sh` describes every
configuration; `DRY=1` prints the commands without running them):

```bash
bash run_cage_ablation.sh design  restaurant hotel phone education beauty   # hard gate, no gate, free embedding
bash run_cage_ablation.sh review  restaurant hotel phone education beauty   # replacing cross-attention, minimal CAGE
bash run_cage_ablation.sh components restaurant                             # loss balancing, ACD loss, Child-Tuning
bash run_cage_ablation.sh heads   restaurant
bash run_cage_ablation.sh adapter restaurant
```

**Baselines.** `scripts/kaggle_queue.py` holds the exact command of every baseline run (BERT-based, CNN/BiLSTM-CNN,
T5-based and instruction-tuned) and splits them into GPU sessions; `python3 scripts/kaggle_queue.py list` shows what
is still missing from `outputs/`. Single runs can be launched directly, for example:

```bash
D="--train_path Hotel_ABSA/Train.txt --dev_path Hotel_ABSA/Dev.txt --test_path Hotel_ABSA/Test.txt"
python3 baselines/bert_baseline.py --mode train $D --seeds 42 --output_dir outputs/bert_hotel_phobert/seed_42 \
    --model_name vinai/phobert-base-v2 --segmenter pyvi
python3 baselines/t5_instruction_tuning.py --domain hotel --format nl --lang vi $D --seeds 42 \
    --output_dir outputs/t5_hotel_nl_vi/seed_42
```

**LLM prompting baselines** (`llm_baseline/`) call a hosted gpt-oss-120b model; put the credentials read by
`llm_preprocessing/llm_client.py` in a `.env` file first.

**Statistics and tables.**

```bash
python3 scripts/cage_stats.py              # main results, significance tests, fixed vs learned, name vs description
python3 scripts/ablation_effects.py        # paired differences, 95% CIs and Holm correction for every ablation
python3 scripts/query_swap_stats.py        # counterfactual query substitution
python3 scripts/category_analysis.py       # per-category gains over PhoBERT
python3 scripts/category_gain_controls.py  # gains controlled for domain and support, description similarity
python3 scripts/subtask_breakdown.py outputs/cage_v2_hotel_fixed/seed_*/test_predictions.jsonl  # category vs polarity errors
```

**Training cost and latency.** `bash run_kaggle_profile.sh` trains every model for one epoch on Hotel with its own
configuration and records time per epoch, peak GPU memory, parameter count and per-review inference latency
(`cost_profile.py` defines each measurement).

## Layout

| Path | Contents |
|---|---|
| `model.py`, `train_mtl_acsa_v2.py` | CAGE model and trainer (all ablation switches are command-line flags) |
| `mapper.py` | category descriptions and names per domain |
| `evaluate.py`, `infer_v2.py` | per-category evaluation; inference from a saved checkpoint |
| `baselines/` | BERT-based, CNN/BiLSTM-CNN, SVM, T5 and instruction-tuned baselines |
| `llm_baseline/` | LLM prompting baselines (zero-shot, 6-agent, retrieval in-context learning, kNN control) |
| `scripts/` | statistics, analyses and the run queue |
| `run_*.sh` | launchers for the runs above |
