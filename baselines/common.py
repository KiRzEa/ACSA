#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for every script in baselines/: block-format data loading
(reused from llm_preprocessing, kept dependency-light), category inference,
prediction I/O, and the joint (category, polarity) micro-P/R/F1 metric used
throughout the paper (Section 4.2) -- pooling TP/FP/FN across every label
before computing precision/recall/F1 once, not averaging per-category scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_preprocessing.data_io import Example, load_examples  # noqa: E402

__all__ = [
    "Example",
    "load_examples",
    "infer_categories",
    "micro_prf",
    "write_predictions",
    "write_metrics",
    "SENTIMENTS",
]

SENTIMENTS = ["positive", "neutral", "negative"]


def infer_categories(*splits: Sequence[Example]) -> List[str]:
    """Sorted union of every category seen across the given splits. Mirrors
    train_mtl_acsa_v2.infer_categories so baselines see the same label space
    as the main architecture -- categories are a data fact, not a hardcoded
    per-domain constant (mapper.py's CATEGORY_DESCRIPTIONS holds NL glosses
    for those same categories, not the canonical label set itself)."""
    cats = set()
    for split in splits:
        for ex in split:
            for c, _ in ex.labels:
                cats.add(c)
    return sorted(cats)


def micro_prf(
    gold: List[List[Tuple[str, str]]], pred: List[List[Tuple[str, str]]]
) -> Dict[str, float]:
    """Micro-averaged P/R/F1 over the joint (category, polarity) label space,
    pooled across all examples and categories -- the exact metric described
    in the paper's Evaluation Metrics section."""
    tp = fp = fn = 0
    for g_pairs, p_pairs in zip(gold, pred):
        g_set = set(g_pairs)
        p_set = set(p_pairs)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def write_predictions(
    path: str | Path,
    examples: Sequence[Example],
    pred_labels: List[List[Tuple[str, str]]],
) -> None:
    """Write one JSON object per line: {"id", "text", "gold", "prediction"}.
    Schema matches evaluate.py's evaluate_jsonl_per_category (gold/prediction
    as lists of {"category", "sentiment"}), so the existing per-category
    analysis tooling works unchanged on every baseline's output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex, preds in zip(examples, pred_labels):
            record = {
                "id": ex.sample_id,
                "text": ex.text,
                "gold": [{"category": c, "sentiment": s} for c, s in ex.labels],
                "prediction": [{"category": c, "sentiment": s} for c, s in preds],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metrics(path: str | Path, metrics: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
