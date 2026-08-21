#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble several train_mtl_acsa.py checkpoints for the same domain by
averaging their logits, then fuse + threshold as usual. A "free" way to
combine complementary models -- no additional GPU training, just inference
passes over checkpoints you already have. Checkpoints may differ in loss
function, sampler, or architecture hyperparameters (num_attention_heads,
adapter_dim) -- each is rebuilt from its own saved args, so this works across
today's Child-Tuning / ASL / focal / oversampling / architecture-sweep runs.

Example:
    python ensemble.py \\
        --checkpoints outputs/phobert_mtl_acsa_hotel/best_model.pt \\
                      outputs/phobert_mtl_acsa_hotel_childtuning/best_model.pt \\
                      outputs/phobert_mtl_acsa_hotel_asl/best_model.pt \\
        --output_dir outputs/ensemble_hotel --per_category_threshold
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

import infer as I
import mapper
import train_mtl_acsa as T

logger = logging.getLogger("ensemble")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _collect_for_checkpoint(
    checkpoint_path: Path, split: str, batch_size_override: int | None, device: torch.device
) -> Dict[str, np.ndarray]:
    """Load one checkpoint, run it over `split`, return collect_outputs' raw
    dict, then free the model before the caller loads the next checkpoint."""
    checkpoint = I._load_checkpoint(checkpoint_path)
    train_args = argparse.Namespace(**checkpoint["args"])
    categories = checkpoint["categories"]

    domain_arg = train_args.domain or "restaurant"
    domain_key = T.DOMAIN_KEY[domain_arg]
    is_raw = domain_arg in T.RAW_TEXT_DOMAIN_ROOTS
    category_descriptions = mapper.CATEGORY_DESCRIPTIONS.get(domain_key, T.CATEGORY_DESCRIPTIONS_VI)

    split_path = I._resolve_split_path(train_args, domain_key, is_raw, split, None)
    examples = I._load_split(domain_key, is_raw, split_path)

    seg_name = "pyvi" if is_raw else "none"
    segmenter = T.build_segmenter(seg_name)
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name, use_fast=False)
    batch_size = batch_size_override or train_args.eval_batch_size
    loader = I._build_loader(examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)

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
        category_self_attention=getattr(train_args, "category_self_attention", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    raw = T.collect_outputs(model, loader, device)
    raw["_categories"] = categories
    raw["_domain_key"] = domain_key

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return raw


def _average_logits(all_raw: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Average acd/sent/joint logits across checkpoints; everything else
    (labels, ids, text) is a data fact, identical across checkpoints, so just
    taken from the first one."""
    averaged = dict(all_raw[0])
    for key in ("acd_logits", "sent_logits", "joint_logits"):
        stacked = np.stack([raw[key] for raw in all_raw], axis=0)
        averaged[key] = stacked.mean(axis=0)
    return averaged


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Device: %s", device)

    checkpoint_paths = [Path(p) for p in args.checkpoints]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def collect_all(split: str) -> List[Dict[str, np.ndarray]]:
        results = []
        for i, ckpt_path in enumerate(checkpoint_paths, start=1):
            logger.info("[%d/%d] %s split: %s", i, len(checkpoint_paths), split, ckpt_path)
            raw = _collect_for_checkpoint(ckpt_path, split, args.batch_size, device)
            results.append(raw)
        categories_sets = {tuple(r["_categories"]) for r in results}
        if len(categories_sets) > 1:
            raise ValueError(
                f"Checkpoints have different category sets, can't ensemble: {categories_sets}"
            )
        domain_keys = {r["_domain_key"] for r in results}
        if len(domain_keys) > 1:
            raise ValueError(f"Checkpoints are for different domains, can't ensemble: {domain_keys}")
        return results

    dev_raw_list = collect_all("dev")
    dev_ensemble = _average_logits(dev_raw_list)
    categories = dev_raw_list[0]["_categories"]

    if args.per_category_threshold:
        threshold = T.tune_thresholds_per_category(
            dev_ensemble, len(categories), min_support=args.threshold_min_support
        )
        logger.info(
            "Per-category thresholds tuned on ensembled dev (min_support=%d): mean=%.2f",
            args.threshold_min_support, float(threshold.mean()),
        )
    else:
        threshold, dev_metrics, _ = T.tune_threshold(dev_ensemble)
        logger.info("Global threshold tuned on ensembled dev: %.2f (dev micro-F1=%.4f)",
                    threshold, dev_metrics["acsa_f1_micro"])

    test_raw_list = collect_all(args.split)
    test_ensemble = _average_logits(test_raw_list)

    metrics, pred = T.compute_metrics(test_ensemble, threshold)

    pred_path = output_dir / f"{args.split}_predictions.jsonl"
    T.write_predictions(pred_path, test_ensemble, pred, categories)
    metrics_out = dict(metrics)
    metrics_out["threshold_full"] = threshold.tolist() if isinstance(threshold, np.ndarray) else threshold
    metrics_out["checkpoints"] = [str(p) for p in checkpoint_paths]
    metrics_path = output_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Predictions written to: %s", pred_path)
    logger.info("Metrics written to: %s", metrics_path)
    logger.info("=== ENSEMBLE %s METRICS (%d checkpoints) ===", args.split.upper(), len(checkpoint_paths))
    for k, v in metrics.items():
        logger.info("%-32s: %.6f", k, v)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ensemble multiple train_mtl_acsa.py checkpoints by averaging logits")
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Paths to two or more best_model.pt files")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=None, help="Defaults to each checkpoint's own eval_batch_size")
    p.add_argument("--per_category_threshold", action="store_true", default=True, dest="per_category_threshold")
    p.add_argument("--no_per_category_threshold", action="store_false", dest="per_category_threshold")
    p.add_argument("--threshold_min_support", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("Need at least 2 --checkpoints to ensemble")
    run(args)
