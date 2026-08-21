#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a train_mtl_acsa.py checkpoint against a held-out split (dev or test) and
write predictions + metrics, without retraining. Everything about the run
(domain, model_name, architecture hyperparameters, category set) is read back
from the checkpoint itself -- only --checkpoint is required.

By default reuses the threshold the checkpoint was saved with. Pass
--per_category_threshold to instead tune per-category thresholds fresh on dev
and apply them to --split -- this lets you A/B threshold strategies against an
already-trained model without retraining, since threshold selection is pure
post-hoc reranking of already-collected logits.

Examples:
    python infer.py --checkpoint outputs/phobert_mtl_acsa_hotel/best_model.pt --split test
    python infer.py --checkpoint outputs/phobert_mtl_acsa_hotel/best_model.pt --split test \\
        --per_category_threshold --threshold_min_support 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import mapper
import train_mtl_acsa as T

logger = logging.getLogger("infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _load_checkpoint(checkpoint_path: Path) -> dict:
    # weights_only=False: this checkpoint was written by train_mtl_acsa.py's own
    # save_checkpoint(), not an untrusted source -- required since PyTorch >=2.6
    # defaults torch.load to weights_only=True, which can't unpickle the numpy
    # array stored here when the run used --per_category_threshold.
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_split_path(
    train_args: argparse.Namespace, domain_key: str, is_raw: bool, split_name: str, override: str | None
) -> str:
    if override:
        return override
    if is_raw:
        raw_root = Path(train_args.raw_root) if train_args.raw_root else (
            Path(T.__file__).resolve().parent / T.RAW_TEXT_DOMAIN_ROOTS[train_args.domain or "restaurant"]
        )
        return str(raw_root / f"{split_name.capitalize()}.txt")
    pair_root = Path(train_args.absa_llms_root) / domain_key
    return str(pair_root / f"{split_name.capitalize()}.csv")


def _load_split(domain_key: str, is_raw: bool, path: str) -> List[T.Example]:
    return T.parse_dataset(path) if is_raw else T.load_pair_split(domain_key, path)


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

    domain_arg = train_args.domain or "restaurant"
    domain_key = T.DOMAIN_KEY[domain_arg]
    is_raw = domain_arg in T.RAW_TEXT_DOMAIN_ROOTS
    category_descriptions = mapper.CATEGORY_DESCRIPTIONS.get(domain_key, T.CATEGORY_DESCRIPTIONS_VI)
    logger.info("Domain: %s (categories=%d)", domain_arg, len(categories))

    batch_size = args.batch_size or train_args.eval_batch_size
    seg_name = "pyvi" if is_raw else "none"
    segmenter = T.build_segmenter(seg_name)
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name, use_fast=False)

    category_texts = [
        segmenter(category_descriptions.get(c, c.replace("#", " ").replace("&", " và ")))
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
        # getattr, not train_args.category_self_attention: checkpoints saved before
        # this flag existed won't have it in their saved args at all.
        category_self_attention=getattr(train_args, "category_self_attention", False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Device: %s", device)

    target_path = _resolve_split_path(train_args, domain_key, is_raw, args.split, args.data_path)
    target_examples = _load_split(domain_key, is_raw, target_path)
    logger.info("Loaded %d examples from %s (--split %s)", len(target_examples), target_path, args.split)
    target_loader = _build_loader(target_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)
    target_raw = T.collect_outputs(model, target_loader, device)

    if args.per_category_threshold:
        if args.split == "dev":
            dev_raw = target_raw
        else:
            dev_path = _resolve_split_path(train_args, domain_key, is_raw, "dev", args.dev_data_path)
            dev_examples = _load_split(domain_key, is_raw, dev_path)
            logger.info("Loaded %d dev examples from %s for threshold tuning", len(dev_examples), dev_path)
            dev_loader = _build_loader(dev_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)
            dev_raw = T.collect_outputs(model, dev_loader, device)
        threshold = T.tune_thresholds_per_category(
            dev_raw, len(categories), min_support=args.threshold_min_support
        )
        logger.info(
            "Per-category thresholds tuned on dev (min_support=%d): mean=%.2f",
            args.threshold_min_support, float(threshold.mean()),
        )
    else:
        threshold = checkpoint["threshold"]
        logger.info(
            "Using checkpoint's saved threshold: %s",
            "per-category" if isinstance(threshold, np.ndarray) else f"{threshold:.2f}",
        )

    metrics, pred = T.compute_metrics(target_raw, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_pct" if args.per_category_threshold else ""
    pred_path = output_dir / f"{args.split}{suffix}_predictions.jsonl"
    T.write_predictions(pred_path, target_raw, pred, categories)
    metrics_out = dict(metrics)
    metrics_out["threshold_full"] = threshold.tolist() if isinstance(threshold, np.ndarray) else threshold
    metrics_path = output_dir / f"{args.split}{suffix}_inference_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Predictions written to: %s", pred_path)
    logger.info("Metrics written to: %s", metrics_path)
    logger.info("=== %s METRICS ===", args.split.upper())
    for k, v in metrics.items():
        logger.info("%-32s: %.6f", k, v)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a trained train_mtl_acsa.py checkpoint on a held-out split")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument(
        "--data_path", type=str, default=None,
        help="Overrides the --split path the checkpoint's domain would otherwise resolve to",
    )
    p.add_argument(
        "--dev_data_path", type=str, default=None,
        help="Overrides the dev-split path used for --per_category_threshold tuning when --split test",
    )
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to the checkpoint's own directory")
    p.add_argument("--batch_size", type=int, default=None, help="Defaults to the value used at training time")
    p.add_argument(
        "--per_category_threshold", action="store_true",
        help="Tune per-category thresholds fresh on dev and apply to --split, instead of reusing the "
        "checkpoint's saved threshold. No retraining -- lets you A/B threshold strategies on an "
        "already-trained model.",
    )
    p.add_argument(
        "--threshold_min_support", type=int, default=20,
        help="Only used with --per_category_threshold; see train_mtl_acsa.py's flag of the same name.",
    )
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
