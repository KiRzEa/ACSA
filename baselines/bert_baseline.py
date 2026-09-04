#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT-based baseline (the "PhoBERT" / "Ensemble BERTs" rows in Tables 3/4),
ported from notebooks/legacy/bert_baseline_prototype.ipynb: a single shared
encoder, [CLS] pooling, dropout, and one 4-way softmax head per category
(NONE / positive / neutral / negative) -- summed per-category cross-entropy,
same joint-metric evaluation as every other baseline in baselines/. Reimplemented
in PyTorch (the prototype used Keras/TF) so it shares model.py's tokenizer/
AutoModel conventions.

Modes:
  --mode train    Fine-tune one encoder (e.g. vinai/phobert-base-v2 or
                   xlm-roberta-base) and write test_predictions.jsonl /
                   multi_seed_summary.json, best of --seeds.
  --mode ensemble Average softmax probabilities from >=2 already-trained
                   checkpoints (e.g. a PhoBERT run + an XLM-R run) -- the
                   "Ensemble BERTs" row. Checkpoints must share the same
                   category list (i.e. same domain).

Segmentation: PhoBERT-family models expect pyvi word-segmented input;
multilingual models (xlm-roberta-*) expect raw text -- pass --segmenter
accordingly, mirroring train_mtl_acsa_v2.py's --segmenter flag.

Examples:
    python3 baselines/bert_baseline.py --mode train \\
        --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/bert_beauty_phobert \\
        --model_name vinai/phobert-base-v2 --segmenter pyvi --seeds 42,123,2024

    python3 baselines/bert_baseline.py --mode train \\
        --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/bert_beauty_xlmr \\
        --model_name xlm-roberta-base --segmenter none --seeds 42,123,2024

    python3 baselines/bert_baseline.py --mode ensemble \\
        --checkpoints outputs/bert_beauty_phobert/best_model.pt outputs/bert_beauty_xlmr/best_model.pt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/bert_beauty_ensemble
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvi import ViTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Example, infer_categories, load_examples, micro_prf, write_metrics, write_predictions  # noqa: E402

logger = logging.getLogger("bert_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

LABELS = ["NONE", "positive", "neutral", "negative"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_segmenter(name: str):
    if name == "pyvi":
        return ViTokenizer.tokenize
    return lambda text: text


class ACSADataset(Dataset):
    def __init__(self, examples: Sequence[Example], categories: List[str], tokenizer, segmenter, max_length: int):
        self.examples = examples
        self.categories = categories
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            self.segmenter(ex.text), truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        label_by_cat = {c: s for c, s in ex.labels}
        targets = [LABEL2IDX.get(label_by_cat.get(cat, "NONE"), 0) for cat in self.categories]
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(targets, dtype=torch.long)


class BertBaseline(nn.Module):
    def __init__(self, model_name: str, num_categories: int, dropout: float = 0.25):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, len(LABELS)) for _ in range(num_categories)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return [head(cls) for head in self.heads]


def logits_to_predictions(all_logits: List[np.ndarray], categories: List[str]) -> List[List[Tuple[str, str]]]:
    """all_logits: per-category array of shape (N, len(LABELS)), already
    softmax-averaged if ensembling. Returns argmax predictions per example."""
    n = all_logits[0].shape[0]
    predictions: List[List[Tuple[str, str]]] = [[] for _ in range(n)]
    for c, cat in enumerate(categories):
        pred_idx = all_logits[c].argmax(axis=1)
        for i, p in enumerate(pred_idx):
            if p != 0:
                predictions[i].append((cat, LABELS[p]))
    return predictions


def collect_probs(model: BertBaseline, loader: DataLoader, device: torch.device, num_categories: int) -> List[np.ndarray]:
    model.eval()
    per_cat_probs = [[] for _ in range(num_categories)]
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            for c in range(num_categories):
                per_cat_probs[c].append(F.softmax(logits[c], dim=1).cpu().numpy())
    return [np.concatenate(p, axis=0) for p in per_cat_probs]


def train_one_seed(
    train: List[Example], dev: List[Example], test: List[Example], categories: List[str], args: argparse.Namespace, seed: int
) -> Tuple[Dict[str, float], List[List[Tuple[str, str]]], Dict]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    segmenter = build_segmenter(args.segmenter)

    train_ds = ACSADataset(train, categories, tokenizer, segmenter, args.max_length)
    dev_ds = ACSADataset(dev, categories, tokenizer, segmenter, args.max_length)
    test_ds = ACSADataset(test, categories, tokenizer, segmenter, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size)

    model = BertBaseline(args.model_name, len(categories), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def evaluate(loader: DataLoader, examples: List[Example]):
        probs = collect_probs(model, loader, device, len(categories))
        predictions = logits_to_predictions(probs, categories)
        metrics = micro_prf([ex.labels for ex in examples], predictions)
        return metrics, predictions, probs

    best_dev_f1, best_state, patience_left = -1.0, None, args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, y in train_loader:
            input_ids, attention_mask, y = input_ids.to(device), attention_mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = sum(F.cross_entropy(logits[c], y[:, c]) for c in range(len(categories)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        dev_metrics, _, _ = evaluate(dev_loader, dev)
        logger.info(
            "[seed %d] epoch %d/%d train_loss=%.4f dev_f1=%.2f",
            seed, epoch, args.epochs, total_loss / max(1, len(train_loader)), dev_metrics["f1"],
        )
        if dev_metrics["f1"] > best_dev_f1 + args.min_delta:
            best_dev_f1 = dev_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[seed %d] early stopping at epoch %d (best dev_f1=%.2f)", seed, epoch, best_dev_f1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_predictions, _ = evaluate(test_loader, test)

    checkpoint = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_name": args.model_name,
        "categories": categories,
        "segmenter": args.segmenter,
        "max_length": args.max_length,
        "dropout": args.dropout,
    }
    return test_metrics, test_predictions, checkpoint


def run_train(args: argparse.Namespace) -> None:
    train = load_examples(args.train_path)
    dev = load_examples(args.dev_path)
    test = load_examples(args.test_path)
    categories = infer_categories(train, dev, test)
    logger.info("Loaded %d train / %d dev / %d test examples, %d categories", len(train), len(dev), len(test), len(categories))

    output_dir = Path(args.output_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed_results = []
    best_checkpoint = None
    for seed in seeds:
        metrics, predictions, checkpoint = train_one_seed(train, dev, test, categories, args, seed)
        logger.info("[seed %d] test P/R/F1: %.2f / %.2f / %.2f", seed, metrics["precision"], metrics["recall"], metrics["f1"])
        per_seed_results.append({"seed": seed, "metrics": metrics, "predictions": predictions})
        if best_checkpoint is None or metrics["f1"] > best_checkpoint["metrics"]["f1"]:
            best_checkpoint = {"metrics": metrics, "state": checkpoint}

    best = max(per_seed_results, key=lambda r: r["metrics"]["f1"])
    logger.info("Best of %d seeds: seed=%d test_f1=%.2f", len(seeds), best["seed"], best["metrics"]["f1"])

    write_predictions(output_dir / "test_predictions.jsonl", test, best["predictions"])
    summary = {
        "best_seed": best["seed"],
        "best": best["metrics"],
        "per_seed": [{"seed": r["seed"], **r["metrics"]} for r in per_seed_results],
        "categories": categories,
        "model_name": args.model_name,
    }
    write_metrics(output_dir / "multi_seed_summary.json", summary)
    if args.save_checkpoint:
        torch.save(best_checkpoint["state"], output_dir / "best_model.pt")
        logger.info("Saved best checkpoint to %s", output_dir / "best_model.pt")
    logger.info("Wrote predictions/summary to %s", output_dir)


def run_ensemble(args: argparse.Namespace) -> None:
    if len(args.checkpoints) < 2:
        raise ValueError("--mode ensemble needs at least 2 --checkpoints")
    test = load_examples(args.test_path)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    all_probs: List[List[np.ndarray]] = []
    categories = None
    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if categories is None:
            categories = ckpt["categories"]
        elif ckpt["categories"] != categories:
            raise ValueError(f"{ckpt_path} has a different category list -- checkpoints must be for the same domain")

        tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"], use_fast=False)
        segmenter = build_segmenter(ckpt["segmenter"])
        test_ds = ACSADataset(test, categories, tokenizer, segmenter, ckpt["max_length"])
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size)

        model = BertBaseline(ckpt["model_name"], len(categories), dropout=ckpt["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        probs = collect_probs(model, test_loader, device, len(categories))
        all_probs.append(probs)
        logger.info("Collected softmax probs from %s (%s)", ckpt_path, ckpt["model_name"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    num_categories = len(categories)
    averaged = [np.mean([probs[c] for probs in all_probs], axis=0) for c in range(num_categories)]
    predictions = logits_to_predictions(averaged, categories)
    metrics = micro_prf([ex.labels for ex in test], predictions)
    logger.info("Ensemble test P/R/F1: %.2f / %.2f / %.2f", metrics["precision"], metrics["recall"], metrics["f1"])

    output_dir = Path(args.output_dir)
    write_predictions(output_dir / "test_predictions.jsonl", test, predictions)
    write_metrics(output_dir / "test_metrics.json", {**metrics, "checkpoints": [str(p) for p in args.checkpoints], "categories": categories})
    logger.info("Wrote predictions/metrics to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tuned BERT-family ACSA baseline (per-category CLS classification)")
    p.add_argument("--mode", choices=["train", "ensemble"], default="train")
    p.add_argument("--train_path", type=str)
    p.add_argument("--dev_path", type=str)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    p.add_argument("--segmenter", choices=["none", "pyvi"], default="pyvi")
    p.add_argument("--seeds", type=str, default="42,123,2024")
    p.add_argument("--checkpoints", type=str, nargs="+", help="--mode ensemble: >=2 best_model.pt paths")
    p.add_argument("--max_length", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--save_checkpoint", action="store_true", default=True)
    p.add_argument("--no_save_checkpoint", action="store_false", dest="save_checkpoint")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.mode == "train":
        if not args.train_path or not args.dev_path:
            raise ValueError("--mode train needs --train_path and --dev_path")
        run_train(args)
    else:
        run_ensemble(args)
