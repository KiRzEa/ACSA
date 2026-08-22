#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a train_mtl_acsa_v2.py checkpoint against a held-out split (dev or test)
and write predictions + metrics, without retraining. Everything about the run
(paths, model_name, architecture hyperparameters, category set, extra vocab)
is read back from the checkpoint and its output_dir -- only --checkpoint is
required.

By default reuses the threshold the checkpoint was saved with. Pass
--per_category_threshold to instead tune per-category thresholds fresh on dev
and apply them to --split.

Examples:
    python infer_v2.py --checkpoint outputs/phobert_mtl_acsa_hotel_v2_childtuning/best_model.pt --split test
    python infer_v2.py --checkpoint outputs/.../best_model.pt --split test --per_category_threshold
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import train_mtl_acsa_v2 as T
from evaluate import evaluate_jsonl_per_category

logger = logging.getLogger("infer_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _load_checkpoint(checkpoint_path: Path) -> dict:
    # weights_only=False: written by train_mtl_acsa_v2.py's own save_checkpoint(),
    # not an untrusted source -- required since PyTorch >=2.6 defaults
    # torch.load to weights_only=True, which can't unpickle numpy arrays
    # (stored here when the run used --per_category_threshold).
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_extra_vocab(checkpoint_path: Path, train_args: argparse.Namespace) -> Optional[List[str]]:
    """Prefer extra_vocab.json saved next to the checkpoint at train time
    (exact, self-contained) over re-reading --extra_vocab_file's original
    path (may not exist / may have changed in this environment)."""
    saved = checkpoint_path.parent / "extra_vocab.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    if getattr(train_args, "extra_vocab_file", None):
        path = Path(train_args.extra_vocab_file)
        if path.exists():
            return [
                line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        logger.warning(
            "train_args.extra_vocab_file=%s not found and no extra_vocab.json next to checkpoint -- "
            "proceeding WITHOUT vocab extension, tokenization will not match training exactly.",
            train_args.extra_vocab_file,
        )
    return None


def _build_loader(examples, categories, tokenizer, segmenter, max_length, batch_size) -> DataLoader:
    dataset = T.ACSADataset(examples, categories)
    collate = T.make_collate_fn(tokenizer, segmenter, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def run(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    train_args = argparse.Namespace(**checkpoint["args"])
    categories = checkpoint["categories"]
    logger.info("Categories: %d", len(categories))

    batch_size = args.batch_size or train_args.eval_batch_size
    segmenter = T.build_segmenter(train_args.segmenter)
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name, use_fast=False)

    extra_vocab = _resolve_extra_vocab(checkpoint_path, train_args)
    if extra_vocab:
        logger.info("Reconstructing extended vocab: %d tokens", len(extra_vocab))

    category_texts = [
        segmenter(T.CATEGORY_DESCRIPTIONS_VI.get(c, c.replace("#", " ").replace("&", " và ")))
        for c in categories
    ]
    model = T.CategoryConditionedMTL(
        model_name=train_args.model_name,
        tokenizer=tokenizer,
        categories=categories,
        category_texts=category_texts,
        num_attention_heads=train_args.num_attention_heads,
        adapter_dim=train_args.adapter_dim,
        dropout=train_args.dropout,
        gradient_checkpointing=False,
        extra_vocab=extra_vocab,
        # getattr, not train_args.entity_attribute_heads: checkpoints saved
        # before this flag existed won't have it in their saved args at all.
        entity_attribute_heads=getattr(train_args, "entity_attribute_heads", False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Device: %s", device)

    split_path = args.data_path or getattr(train_args, f"{args.split}_path")
    target_examples = T.parse_dataset(split_path)
    logger.info("Loaded %d examples from %s (--split %s)", len(target_examples), split_path, args.split)
    target_loader = _build_loader(target_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)
    target_raw = T.collect_outputs(model, target_loader, device)

    if args.per_category_threshold:
        dev_examples = T.parse_dataset(args.data_path if args.split == "dev" and args.data_path else train_args.dev_path)
        dev_loader = _build_loader(dev_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)
        dev_raw = T.collect_outputs(model, dev_loader, device)
        threshold, dev_metrics, _ = T.tune_threshold(dev_raw)
        logger.info("Re-tuned global threshold on dev: %.2f (dev micro-F1=%.4f)", threshold, dev_metrics["acsa_f1_micro"])
    else:
        threshold = float(checkpoint["threshold"])
        logger.info("Using checkpoint's saved threshold: %.2f", threshold)

    metrics, pred = T.compute_metrics(target_raw, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.split}_predictions.jsonl"
    T.write_predictions(pred_path, target_raw, pred, categories)
    metrics_path = output_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Predictions written to: %s", pred_path)
    logger.info("Metrics written to: %s", metrics_path)
    logger.info("=== %s METRICS ===", args.split.upper())
    for k, v in metrics.items():
        logger.info("%-32s: %.6f", k, v)
    evaluate_jsonl_per_category(pred_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference with a saved train_mtl_acsa_v2.py checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--data_path", type=str, default=None, help="Override the split path saved in the checkpoint's args")
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to the checkpoint's own directory")
    p.add_argument("--batch_size", type=int, default=None, help="Defaults to the checkpoint's eval_batch_size")
    p.add_argument("--per_category_threshold", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
