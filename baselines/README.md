# baselines/

Clean, script-based (not notebook) implementations of every row in Tables 3/4
that isn't our own architecture or a specific external paper's unreproducible
numbers -- SVM, CNN, BiLSTM-CNN, BERT-based, and instruction-tuned
small-language-model baselines -- generalized to run on any of the 5 domains'
`Train.txt`/`Dev.txt`/`Test.txt` block-format data. Every script writes the
same `test_predictions.jsonl` schema (`{"id", "text", "gold", "prediction"}`,
compatible with `evaluate.py`'s per-category breakdown) and reports the same
joint micro-P/R/F1 metric (`common.micro_prf`) described in the paper's
Evaluation Metrics section.

Each script is ported from one of the prototype notebooks in
`notebooks/legacy/`, generalized from whichever single domain/taxonomy the
prototype was originally built against to any current domain's category list
(inferred from the data itself, `common.infer_categories`), and switched from
this project's earlier scattered notebook environments (Colab + TF/Keras +
`simpletransformers`) to plain PyTorch/HuggingFace `transformers`, matching
`model.py` / `train_mtl_acsa_v2.py`.

| Script | Table row(s) | Ported from | Needs GPU? |
|---|---|---|---|
| `svm_tfidf_baseline.py` | SVM | `baseline_method_paper.pdf` method (TF-IDF + linear SVM) | No -- CPU, seconds |
| `cnn_baseline.py` | CNN | `bilstm_cnn_prototype.ipynb`, BiLSTM branch removed | Yes (fast) |
| `cnn_baseline.py --use_bilstm` | BiLSTM-CNN | `bilstm_cnn_prototype.ipynb`, unmodified architecture | Yes |
| `bert_baseline.py` | XLM-R / PhoBERT / Ensemble BERTs | `bert_baseline_prototype.ipynb` | Yes |
| `t5_seq2seq_baseline.py` | mT5-large / viT5-large / viT5-base ("T5-based Models") | `t5_instruction_prototype.ipynb`'s `simpletransformers.T5Model` "csc" task | Yes |
| `t5_instruction_tuning.py` | NL/Code Instruction-Vi/En (4 rows, "Instruction Tuning with Small Language Models") | `t5_quadruplet_acos_prototype.ipynb` + `t5_instruction_prototype.ipynb` | Yes |

Every one of these was smoke-tested end-to-end locally (real run on real data
for the CPU-only SVM script; tiny-model/short-epoch runs for the rest, using
the *real* default checkpoints -- `vinai/phobert-base-v2`, `xlm-roberta-base`,
`Salesforce/codet5-base`, `VietAI/vit5-base` -- to validate the data
pipeline, training loop, and prediction parsing without a full GPU run)
before being handed off for full training.

**T5-based vs. Instruction Tuning -- what's actually different**: `t5_seq2seq_baseline.py`
passes the model *only the review*, trained purely on (review -> category +
sentiment) pairs with no instruction text at all (`"csc: " + review ->
"CATEGORY: polarity; ..."`); `t5_instruction_tuning.py` wraps the review in
one of 4 explicit instruction prompts (code-style or natural-language, in
Vietnamese or English) telling the model what to extract and how to format
the answer. Same underlying seq2seq mechanics, genuinely different training
signal -- that's why they're separate row groups in the table, and separate
scripts here.

## Why these rows, and not others

- Restaurant/Hotel/Phone (Table 3) already have real numbers for every row
  (published prior work for the statistics-based rows, our own earlier runs
  for BERT-based/T5-based/instruction-tuned) -- nothing here needs re-running
  for those three domains.
- Education's SVM/CNN/BiLSTM-CNN rows already have real published numbers
  (TNUJST5101, the dataset's own paper) -- only its BERT-based, T5-based, and
  instruction-tuned rows are pending; `run_baselines_education.sh` does not
  run `svm_tfidf_baseline.py` or `cnn_baseline.py` for this reason.
- Beauty has no prior published baseline on this exact joint micro-F1 metric
  at all (`docs/references/beauty_dataset_paper.pdf`'s BiGRU+Conv1D result
  is on a different label space -- separate aspect-detection/sentiment F1,
  not joint (category, polarity) F1, and a different 7-aspect single-label
  lipstick taxonomy) -- every row is pending for Beauty, including SVM/CNN/
  BiLSTM-CNN.

## Running

**Kaggle setup**: pin `transformers` to `4.57.6` -- Kaggle's default image at time of
writing ships a newer preview build whose tokenizer-loading refactor fails on
`Salesforce/codet5-base`'s legacy (no `tokenizer.json`) tokenizer with
`TypeError: extra_special_tokens must be a list/tuple of str or AddedToken, ...`
(BERT-family checkpoints are unaffected -- they ship a fast `tokenizer.json`
directly, so they never hit the from-slow conversion path that's broken).
`4.57.6` is confirmed working locally against every model used here
(`vinai/phobert-base-v2`, `xlm-roberta-base`, `Salesforce/codet5-base`,
`VietAI/vit5-base`):
```bash
pip install -q "transformers==4.57.6" pyvi sentencepiece accelerate
```

SVM is fast enough to run locally right now (no Kaggle needed):
```bash
python3 baselines/svm_tfidf_baseline.py \
    --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt --test_path Beauty_ABSA/Test.txt \
    --output_dir outputs/svm_beauty
```

Everything else needs a GPU. `run_baselines_beauty.sh` / `run_baselines_education.sh`
run every pending row for that domain in one idempotent pass (skips any run
whose summary file already exists, so re-running after a Kaggle session gets
cut off at the 12h limit is always safe -- same pattern as
`run_ablation_session1.sh`/`run_ablation_session2.sh` at the repo root):
```bash
bash baselines/run_baselines_beauty.sh
bash baselines/run_baselines_education.sh
```

Each individual script also runs standalone, e.g. one instruction-tuning
variant on Education:
```bash
python3 baselines/t5_instruction_tuning.py \
    --domain "university course evaluation" --format nl --lang vi \
    --train_path Education_ABSA/Train.txt --dev_path Education_ABSA/Dev.txt --test_path Education_ABSA/Test.txt \
    --output_dir outputs/t5_education_nl_vi --seeds 42
```

`--seeds` accepts a comma-separated list and trains one independent run per
seed, reporting best-of-N test F1 in `multi_seed_summary.json` -- the same
mechanism the paper's other results use for best-of-3-seed reporting. Both
orchestration scripts currently default to a single seed (`--seeds 42`) to
keep the remaining Beauty/Education runs affordable; bump back to
`"42,123,2024"` (in the script or on an individual command) if 3-seed
numbers are wanted later.

## After training: pulling numbers into the paper

Read `<output_dir>/multi_seed_summary.json`'s `"best"` field (precision/
recall/f1) for each row, the same way the ablation study's results were
extracted from `outputs/*/multi_seed_summary.json` earlier in this project.
`bert_baseline.py --mode ensemble` writes `test_metrics.json` instead (no
seed loop -- it deterministically averages two already-trained checkpoints).
