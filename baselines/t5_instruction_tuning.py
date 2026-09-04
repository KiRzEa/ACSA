#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruction-tuned small-language-model baseline (the "NL/Code Instruction-Vi/En"
rows in Tables 3/4), generalized from the ACOS-quadruplet prompting in
notebooks/legacy/t5_quadruplet_acos_prototype.ipynb and the natural-language
paraphrase prompting in notebooks/legacy/t5_instruction_prototype.ipynb.

Unlike those two prototypes, this targets ACSA directly: the generation target
is (category, sentiment) pairs only (we have no aspect-term/opinion-term
annotations for these domains, unlike the English ACOS benchmark the first
prototype was built against), and training uses the standard HF
Seq2SeqTrainer encoder-decoder objective (prompt -> target) instead of the
first prototype's causal-LM-style SFTTrainer packing, which does not fit a
T5 encoder-decoder model and is what caused that prototype's
`item.split("model\\n")[1]` IndexError at generation time.

Four format x language variants (`--format {code,nl} --lang {vi,en}`), matching
Table 3's row names. "Vi"/"En" is the language of the *prompt itself*
(docstring / instruction text) -- the review text is always Vietnamese, and
generated sentiment labels are always the canonical positive/neutral/negative
tokens in both languages, so gold-label matching is unaffected by prompt
language:
  - code+en: an English-commented Python-function-style prompt (closest to
    the first prototype's CODING_FORMAT_PROMPTING, reduced to 2 fields).
  - code+vi: the same code-style prompt with Vietnamese identifiers/docstring.
  - nl+en:   a plain English instruction, target "CATEGORY is SENTIMENT; ...".
  - nl+vi:   a plain Vietnamese instruction, target "CATEGORY là SENTIMENT; ...".

Default base model is Salesforce/codet5-base for --format code (matches the
first prototype) and VietAI/vit5-base for --format nl (native Vietnamese
SentencePiece vocabulary, tokenizes the Vietnamese review text far more
efficiently than codet5-base's English/code vocabulary) -- override either
with --model_name.

Example:
    python3 baselines/t5_instruction_tuning.py \\
        --domain beauty --format nl --lang vi \\
        --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/t5_beauty_nl_vi \\
        --seeds 42,123,2024
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Example, infer_categories, load_examples, micro_prf, write_metrics, write_predictions  # noqa: E402

logger = logging.getLogger("t5_instruction_tuning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

DEFAULT_MODEL = {"code": "Salesforce/codet5-base", "nl": "VietAI/vit5-base"}

CODE_TEMPLATE_EN = '''def extract_category_sentiment_list(review):
    """Extract (category, sentiment) pairs from a {domain} review. sentiment is one of: positive, neutral, negative."""
    review = "{review}"
    category_sentiment_list = []
    return category_sentiment_list

# Run an example
review = "{review}"
results = extract_category_sentiment_list(review)
for pair in results:
    print(f"category_sentiment_list.append({{pair}})")'''

CODE_TEMPLATE_VI = '''def trich_xuat_danh_sach_khia_canh_cam_xuc(danh_gia):
    """Trich xuat cac cap (khia canh, cam xuc) tu mot danh gia {domain}. cam_xuc la mot trong: positive, neutral, negative."""
    danh_gia = "{review}"
    danh_sach_khia_canh_cam_xuc = []
    return danh_sach_khia_canh_cam_xuc

# Chay thu vi du
danh_gia = "{review}"
ket_qua = trich_xuat_danh_sach_khia_canh_cam_xuc(danh_gia)
for cap in ket_qua:
    print(f"danh_sach_khia_canh_cam_xuc.append({{cap}})")'''

NL_TEMPLATE_EN = (
    'Extract every (category, sentiment) pair expressed in the following Vietnamese {domain} review, '
    'where sentiment is one of positive, neutral, or negative.\n'
    'Review: "{review}"\n'
    "Answer:"
)

NL_TEMPLATE_VI = (
    "Hay trich xuat tat ca cac cap (khia canh, cam xuc) duoc the hien trong danh gia {domain} tieng Viet sau day, "
    "voi cam xuc la mot trong ba gia tri: positive, neutral, negative.\n"
    'Danh gia: "{review}"\n'
    "Tra loi:"
)

_CODE_PAIR_RE = re.compile(r'\{[^{}]*?"[^"]+"\s*:\s*"([^"]*)"\s*,\s*"[^"]+"\s*:\s*"([^"]*)"[^{}]*?\}')


def build_prompt(fmt: str, lang: str, domain: str, review: str) -> str:
    template = {("code", "en"): CODE_TEMPLATE_EN, ("code", "vi"): CODE_TEMPLATE_VI,
                ("nl", "en"): NL_TEMPLATE_EN, ("nl", "vi"): NL_TEMPLATE_VI}[(fmt, lang)]
    return template.format(domain=domain, review=review)


def build_target(fmt: str, lang: str, labels: List[Tuple[str, str]]) -> str:
    if fmt == "code":
        key1, key2 = ("category", "sentiment") if lang == "en" else ("khia_canh", "cam_xuc")
        list_name = "category_sentiment_list" if lang == "en" else "danh_sach_khia_canh_cam_xuc"
        lines = [f'{list_name}.append({{"{key1}": "{c}", "{key2}": "{s}"}})' for c, s in labels]
        return "\n".join(lines) if lines else f"{list_name} = []"
    else:
        joiner = " is " if lang == "en" else " la "
        pairs = [f"{c}{joiner}{s}" for c, s in labels]
        return "; ".join(pairs) if pairs else ("no aspects mentioned" if lang == "en" else "khong co khia canh nao duoc de cap")


def parse_prediction(fmt: str, lang: str, text: str) -> List[Tuple[str, str]]:
    if fmt == "code":
        return [(c.strip(), s.strip().lower()) for c, s in _CODE_PAIR_RE.findall(text)]
    joiner = " is " if lang == "en" else " la "
    pairs = []
    for segment in text.split(";"):
        segment = segment.strip()
        if joiner not in segment:
            continue
        cat, sent = segment.rsplit(joiner, 1)
        pairs.append((cat.strip(), sent.strip().lower()))
    return pairs


class PromptTargetDataset(Dataset):
    def __init__(self, examples: List[Example], fmt: str, lang: str, domain: str, tokenizer, max_source_length: int, max_target_length: int):
        self.examples = examples
        self.fmt, self.lang, self.domain = fmt, lang, domain
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = build_prompt(self.fmt, self.lang, self.domain, ex.text)
        target = build_target(self.fmt, self.lang, ex.labels)
        source_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_source_length)
        target_enc = self.tokenizer(text_target=target, truncation=True, max_length=self.max_target_length)
        return {"input_ids": source_enc["input_ids"], "attention_mask": source_enc["attention_mask"], "labels": target_enc["input_ids"]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BestModelTracker(TrainerCallback):
    """Keeps the best epoch's weights in CPU RAM instead of Trainer's normal
    disk checkpoint (model + optimizer + scheduler state, saved every eval
    even with save_total_limit=1) -- on Kaggle's limited working disk, a
    handful of large checkpoints (mostly optimizer state we never need
    again) exhausted it mid-run ("No space left on device"). Nothing here
    ever touches disk; train_one_seed loads best_state back into the model
    in-process right after trainer.train() returns."""

    def __init__(self):
        self.best_metric: float | None = None
        self.best_state: dict | None = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None:
            return
        metric = metrics.get("eval_loss")
        if metric is None or (self.best_metric is not None and metric >= self.best_metric):
            return
        self.best_metric = metric
        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: List[Example], args: argparse.Namespace, device: torch.device) -> List[List[Tuple[str, str]]]:
    model.eval()
    device = next(model.parameters()).device  # Trainer may have placed the model on a device
    # (e.g. MPS) that --cpu / the caller's own cuda-availability check didn't anticipate.
    predictions: List[List[Tuple[str, str]]] = []
    for i in range(0, len(examples), args.eval_batch_size):
        batch = examples[i:i + args.eval_batch_size]
        prompts = [build_prompt(args.format, args.lang, args.domain, ex.text) for ex in batch]
        enc = tokenizer(prompts, truncation=True, max_length=args.max_source_length, padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=args.max_target_length, num_beams=args.num_beams)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for text in decoded:
            predictions.append(parse_prediction(args.format, args.lang, text))
    return predictions


def train_one_seed(train: List[Example], dev: List[Example], test: List[Example], args: argparse.Namespace, seed: int, run_dir: Path):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model_name = args.model_name or DEFAULT_MODEL[args.format]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    train_ds = PromptTargetDataset(train, args.format, args.lang, args.domain, tokenizer, args.max_source_length, args.max_target_length)
    dev_ds = PromptTargetDataset(dev, args.format, args.lang, args.domain, tokenizer, args.max_source_length, args.max_target_length)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",  # never write checkpoints to disk -- BestModelTracker keeps the best epoch in CPU RAM instead
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        logging_steps=50,
        seed=seed,
        report_to=[],
        fp16=torch.cuda.is_available() and not args.cpu,
        use_cpu=args.cpu,
    )
    tracker = BestModelTracker()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), tracker],
    )
    trainer.train()

    if tracker.best_state is not None:
        model.load_state_dict(tracker.best_state)
        logger.info("[seed %d] loaded best epoch (eval_loss=%.4f) from CPU RAM", seed, tracker.best_metric)

    predictions = generate_predictions(model, tokenizer, test, args, device)
    metrics = micro_prf([ex.labels for ex in test], predictions)
    return metrics, predictions


def run(args: argparse.Namespace) -> None:
    train = load_examples(args.train_path)
    dev = load_examples(args.dev_path)
    test = load_examples(args.test_path)
    logger.info("Loaded %d train / %d dev / %d test examples (domain=%s, format=%s, lang=%s)",
                len(train), len(dev), len(test), args.domain, args.format, args.lang)

    output_dir = Path(args.output_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed_results = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}" / "trainer_state"
        metrics, predictions = train_one_seed(train, dev, test, args, seed, run_dir)
        logger.info("[seed %d] test P/R/F1: %.2f / %.2f / %.2f", seed, metrics["precision"], metrics["recall"], metrics["f1"])
        per_seed_results.append({"seed": seed, "metrics": metrics, "predictions": predictions})

    best = max(per_seed_results, key=lambda r: r["metrics"]["f1"])
    logger.info("Best of %d seeds: seed=%d test_f1=%.2f", len(seeds), best["seed"], best["metrics"]["f1"])

    write_predictions(output_dir / "test_predictions.jsonl", test, best["predictions"])
    summary = {
        "best_seed": best["seed"],
        "best": best["metrics"],
        "per_seed": [{"seed": r["seed"], **r["metrics"]} for r in per_seed_results],
        "domain": args.domain,
        "format": args.format,
        "lang": args.lang,
        "model_name": args.model_name or DEFAULT_MODEL[args.format],
    }
    write_metrics(output_dir / "multi_seed_summary.json", summary)
    logger.info("Wrote predictions/summary to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Instruction-tuned seq2seq ACSA baseline (4 prompt format x language variants)")
    p.add_argument("--domain", type=str, required=True, help="Domain name used inside the prompt text, e.g. 'beauty product' or 'restaurant'")
    p.add_argument("--format", choices=["code", "nl"], required=True)
    p.add_argument("--lang", choices=["vi", "en"], required=True)
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--dev_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default=None, help="Overrides the format-based default (codet5-base for code, vit5-base for nl)")
    p.add_argument("--seeds", type=str, default="42,123,2024")
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=200)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
