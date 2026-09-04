# baselines/

Clean, script-based (not notebook) implementations of every non-architecture
row in Tables 3/4 of the paper -- SVM, CNN, BERT-based, and instruction-tuned
small-language-model baselines -- generalized to run on any of the 5 domains'
`Train.txt`/`Dev.txt`/`Test.txt` block-format data. All four scripts write
the same `test_predictions.jsonl` schema (`{"id", "text", "gold",
"prediction"}`, compatible with `evaluate.py`'s per-category breakdown) and
report the same joint micro-P/R/F1 metric (`common.micro_prf`) described in
the paper's Evaluation Metrics section.

Each script is ported from one of the prototype notebooks in
`notebooks/legacy/`, generalized from whichever single domain/taxonomy the
prototype was originally built against to any current domain's category list
(inferred from the data itself, `common.infer_categories`), and switched from
this project's earlier scattered notebook environments (Colab + TF/Keras +
`simpletransformers`) to plain PyTorch/HuggingFace `transformers`, matching
`model.py` / `train_mtl_acsa_v2.py`.

| Script | Table row(s) | Ported from | Needs GPU? |
|---|---|---|---|
| `svm_tfidf_baseline.py` | SVM (Statistics) | `notebooks/legacy/` `baseline_method_paper.pdf` method (TF-IDF + linear SVM) | No -- CPU, seconds |
| `cnn_baseline.py` | CNN (Statistics) | `bilstm_cnn_prototype.ipynb`, BiLSTM branch removed | Yes (fast) |
| `bert_baseline.py` | PhoBERT / Ensemble BERTs | `bert_baseline_prototype.ipynb` | Yes |
| `t5_instruction_tuning.py` | NL/Code Instruction-Vi/En (4 rows) | `t5_quadruplet_acos_prototype.ipynb` + `t5_instruction_prototype.ipynb` | Yes |

All four were smoke-tested end-to-end locally (real run on real data for the
CPU-only SVM script; tiny-model/short-epoch runs for the other three, to
validate the data pipeline, training loop, and prediction parsing without a
GPU) before being handed off for full training.

## Why these rows, and not others

- Restaurant/Hotel/Phone (Table 3) already have real numbers for every row
  (published prior work for the statistics-based rows, our own earlier runs
  for BERT-based/T5-based/instruction-tuned) -- nothing here needs re-running
  for those three domains.
- Education's statistics-based row already has real published numbers
  (TNUJST5101, the dataset's own paper) -- only its BERT-based and
  instruction-tuned rows are pending.
- Beauty has no prior published baseline on this exact joint micro-F1 metric
  at all (`docs/references/beauty_dataset_paper.pdf`'s BiGRU+Conv1D result
  is on a different label space -- separate aspect-detection/sentiment F1,
  not joint (category, polarity) F1, and a different 7-aspect single-label
  lipstick taxonomy) -- every row is pending for Beauty.
- The T5-based rows (`viT5_large`/`viT5_base`) in both tables are a specific
  external paper's (`van2023aspect`) numbers, not something we can
  faithfully reproduce without their exact model/code, so they stay `--`
  for Education/Beauty; only the *our-own* "Instruction Tuning with Small
  Language Models" row group is filled in here.

## Running

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
    --output_dir outputs/t5_education_nl_vi --seeds 42,123,2024
```

`--seeds "42,123,2024"` trains 3 independent runs and reports best-of-3 test
F1 in `multi_seed_summary.json`, the same reporting convention as every other
result in the paper.

## After training: pulling numbers into the paper

Read `<output_dir>/multi_seed_summary.json`'s `"best"` field (precision/
recall/f1) for each row, the same way the ablation study's results were
extracted from `outputs/*/multi_seed_summary.json` earlier in this project.
`bert_baseline.py --mode ensemble` writes `test_metrics.json` instead (no
seed loop -- it deterministically averages two already-trained checkpoints).
