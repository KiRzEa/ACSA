#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T5-based baseline (the "T5-based Models" row group in Tables 3/4 --
mT5-large / viT5-large / viT5-base), ported from
notebooks/legacy/t5_instruction_prototype.ipynb's `simpletransformers.T5Model`
"csc" task. The defining difference from t5_instruction_tuning.py's
Instruction Tuning rows: no instruction text at all -- the model is given
only the review, and learns purely from (review -> category+sentiment)
training pairs to produce the right output, matching the prototype's plain
`"csc: " + review -> target` seq2seq formulation.

The prototype's target format was a fixed per-domain dictionary of short
natural-language category phrases joined by " va " (Vietnamese "and") with a
"tot"/"te"/"tam" (good/bad/so-so) polarity word, fuzzy-matched back out of
generated text by substring search -- workable only because Restaurant's 7
categories each had a hand-picked short gloss. That doesn't generalize to
domains whose categories are themselves multi-word Vietnamese phrases
(Education's "Ky nang giang day", Beauty's "Van de khac"), where the
original's implicit assumption of space-only tokenization would make
category and polarity ambiguous. This keeps the same plain-seq2seq,
no-instruction spirit but uses "<category>: <tot|tam|te>" pairs joined by
"; " -- the raw category label (domain-agnostic, no hand-curated phrase
dictionary needed) with an unambiguous separator -- so parsing is exact
across every domain's category-naming convention.

Base model options (matching the prototype's commented-out choices):
google/mt5-base, google/mt5-large, VietAI/vit5-base, VietAI/vit5-large.

Example:
    python3 baselines/t5_seq2seq_baseline.py \\
        --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/vit5base_beauty \\
        --model_name VietAI/vit5-base --seeds 42
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
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Example, load_examples, micro_prf, write_metrics, write_predictions  # noqa: E402

logger = logging.getLogger("t5_seq2seq_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

POLARITY_TO_WORD = {"positive": "tốt", "neutral": "tạm", "negative": "tệ"}
WORD_TO_POLARITY = {v: k for k, v in POLARITY_TO_WORD.items()}
EMPTY_TARGET = "không có khía cạnh nào"
_PAIR_RE = re.compile(r"([^:;]+):\s*(tốt|tạm|tệ)")


def build_input(review: str) -> str:
    return f"csc: {review}"


def build_target(labels: List[Tuple[str, str]]) -> str:
    pairs = [f"{c}: {POLARITY_TO_WORD.get(s, s)}" for c, s in labels]
    return "; ".join(pairs) if pairs else EMPTY_TARGET


def parse_prediction(text: str) -> List[Tuple[str, str]]:
    return [(c.strip(), WORD_TO_POLARITY[w]) for c, w in _PAIR_RE.findall(text)]


class PromptTargetDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_source_length: int, max_target_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        source_enc = self.tokenizer(build_input(ex.text), truncation=True, max_length=self.max_source_length)
        target_enc = self.tokenizer(text_target=build_target(ex.labels), truncation=True, max_length=self.max_target_length)
        return {"input_ids": source_enc["input_ids"], "attention_mask": source_enc["attention_mask"], "labels": target_enc["input_ids"]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: List[Example], args: argparse.Namespace) -> List[List[Tuple[str, str]]]:
    model.eval()
    device = next(model.parameters()).device
    predictions: List[List[Tuple[str, str]]] = []
    for i in range(0, len(examples), args.eval_batch_size):
        batch = examples[i:i + args.eval_batch_size]
        inputs = [build_input(ex.text) for ex in batch]
        enc = tokenizer(inputs, truncation=True, max_length=args.max_source_length, padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=args.max_target_length, num_beams=args.num_beams)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for text in decoded:
            predictions.append(parse_prediction(text))
    return predictions


def train_one_seed(train: List[Example], dev: List[Example], test: List[Example], args: argparse.Namespace, seed: int, run_dir: Path):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.device_map_auto:
        # Naive model-parallel: splits the model's layers across every GPU
        # CUDA_VISIBLE_DEVICES exposes to this process (accelerate's
        # infer_auto_device_map), pooling their memory for models too large
        # for any single one of them -- do NOT pin CUDA_VISIBLE_DEVICES to a
        # single GPU for a --device_map_auto run, it needs all of them
        # visible. Trainer detects model.hf_device_map and switches out of
        # DataParallel/single-device mode automatically.
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        if torch.cuda.is_available() and not args.cpu:
            model = model.to("cuda")
    if args.gradient_checkpointing:
        model.config.use_cache = False  # incompatible with gradient checkpointing; not needed anyway (predict_with_generate=False during training)

    train_ds = PromptTargetDataset(train, tokenizer, args.max_source_length, args.max_target_length)
    dev_ds = PromptTargetDataset(dev, tokenizer, args.max_source_length, args.max_target_length)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        logging_steps=50,
        seed=seed,
        report_to=[],
        fp16=torch.cuda.is_available() and not args.cpu,
        use_cpu=args.cpu,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    predictions = generate_predictions(model, tokenizer, test, args)
    metrics = micro_prf([ex.labels for ex in test], predictions)
    return metrics, predictions


def run(args: argparse.Namespace) -> None:
    train = load_examples(args.train_path)
    dev = load_examples(args.dev_path)
    test = load_examples(args.test_path)
    logger.info("Loaded %d train / %d dev / %d test examples (model=%s)", len(train), len(dev), len(test), args.model_name)

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
        "model_name": args.model_name,
    }
    write_metrics(output_dir / "multi_seed_summary.json", summary)
    logger.info("Wrote predictions/summary to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plain seq2seq T5-family ACSA baseline (review -> category+sentiment, no instruction)")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--dev_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="VietAI/vit5-base",
                    help="google/mt5-base, google/mt5-large, VietAI/vit5-base, or VietAI/vit5-large")
    p.add_argument("--seeds", type=str, default="42,123,2024")
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=150)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Raise this (and lower --batch_size) to shrink peak memory for -large checkpoints while keeping the same effective batch size")
    p.add_argument("--gradient_checkpointing", action="store_true",
                    help="Trade compute for activation memory -- combine with --device_map_auto for mT5-large/viT5-large")
    p.add_argument("--device_map_auto", action="store_true",
                    help="Split the model's layers across every visible GPU (accelerate device_map='auto') instead of loading the whole model onto one -- "
                         "needed when a single GPU's memory is smaller than the model + optimizer state + activations (e.g. mT5-large/viT5-large on a "
                         "~15GB T4: batch_size=1 alone doesn't help, since AdamW's optimizer state for a ~1.2B-parameter model already exceeds 15GB before "
                         "any activations). Do not also pin CUDA_VISIBLE_DEVICES to a single GPU when using this -- it needs every GPU visible.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
