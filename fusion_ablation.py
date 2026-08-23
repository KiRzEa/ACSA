#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validates the "fusion is the bottleneck" hypothesis cheaply: fits a tiny
learned fusion classifier on top of an EXISTING checkpoint's already-trained
head outputs (acd_logits, sent_logits, joint_logits), instead of the fixed
0.5/0.5-weighted formula in fuse_predictions(). No PhoBERT retraining --
trains on Dev, evaluates on Test, and reports both the old fixed-formula
ACSA-F1 and the new learned-fusion ACSA-F1 side by side on the same split.

If this recovers a meaningful chunk of the sent_oracle-vs-acsa gap, it
justifies baking a learned fusion into the model properly (full retrain).
If it doesn't, the bottleneck isn't fusion after all.

    python fusion_ablation.py --checkpoint outputs/.../best_model.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import train_mtl_acsa_v2 as T
from infer_v2 import _load_checkpoint, _resolve_extra_vocab, _build_loader

logger = logging.getLogger("fusion_ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


class LearnedFusion(nn.Module):
    """Tiny MLP: [acd_logit, sent_logits(3), joint_logits(4)] (8-dim) per
    (example, category) -> 4-class joint label (NONE/POS/NEU/NEG). Shared
    across all categories (same 8->4 mapping regardless of which category),
    matching how the existing fixed formula is also category-agnostic."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_fusion_features(raw: dict) -> np.ndarray:
    """[N, K, 8] = concat(acd_logits[...,None], sent_logits, joint_logits)."""
    acd = raw["acd_logits"][..., None]           # [N, K, 1]
    sent = raw["sent_logits"]                     # [N, K, 3]
    joint = raw["joint_logits"]                    # [N, K, 4]
    return np.concatenate([acd, sent, joint], axis=-1).astype(np.float32)  # [N, K, 8]


def acsa_f1_from_pred_joint(gold_joint: np.ndarray, pred_joint: np.ndarray) -> dict:
    gold_pair = T.build_multilabel_pair_matrix(gold_joint)
    pred_pair = T.build_multilabel_pair_matrix(pred_joint)
    return T.classification_metrics(gold_pair, pred_pair)


def run(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    train_args = argparse.Namespace(**checkpoint["args"])
    categories = checkpoint["categories"]
    logger.info("Categories: %d", len(categories))

    segmenter = T.build_segmenter(train_args.segmenter)
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name, use_fast=False)
    extra_vocab = _resolve_extra_vocab(checkpoint_path, train_args)

    category_texts = [
        segmenter(T.CATEGORY_DESCRIPTIONS_VI.get(c, c.replace("#", " ").replace("&", " và ")))
        for c in categories
    ]
    model = T.CategoryConditionedMTL(
        model_name=train_args.model_name, tokenizer=tokenizer, categories=categories,
        category_texts=category_texts, num_attention_heads=train_args.num_attention_heads,
        adapter_dim=train_args.adapter_dim, dropout=train_args.dropout, gradient_checkpointing=False,
        extra_vocab=extra_vocab, entity_attribute_heads=getattr(train_args, "entity_attribute_heads", False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Device: %s", device)

    dev_examples = T.parse_dataset(train_args.dev_path)
    test_examples = T.parse_dataset(train_args.test_path)
    batch_size = args.batch_size or train_args.eval_batch_size
    dev_loader = _build_loader(dev_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)
    test_loader = _build_loader(test_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)

    logger.info("Collecting head outputs (no retraining, just a forward pass)...")
    dev_raw = T.collect_outputs(model, dev_loader, device)
    test_raw = T.collect_outputs(model, test_loader, device)

    # --- baseline: existing fixed 0.5/0.5 formula, threshold tuned on dev ---
    threshold, dev_metrics, _ = T.tune_threshold(dev_raw)
    baseline_metrics, _ = T.compute_metrics(test_raw, threshold)
    logger.info("Baseline (fixed-formula fusion) test acsa_f1_micro: %.4f", baseline_metrics["acsa_f1_micro"])

    # --- learned fusion: tiny MLP trained on Dev's already-collected outputs ---
    dev_X = torch.tensor(build_fusion_features(dev_raw), device=device)   # [N, K, 8]
    dev_y = torch.tensor(dev_raw["joint_labels"], dtype=torch.long, device=device)  # [N, K]
    test_X = torch.tensor(build_fusion_features(test_raw), device=device)
    test_y = test_raw["joint_labels"].astype(np.int64)

    fusion = LearnedFusion().to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-3)
    n, k, _ = dev_X.shape
    flat_X, flat_y = dev_X.reshape(-1, 8), dev_y.reshape(-1)
    # Class-imbalance weight: NONE vastly outnumbers the 3 present classes,
    # same imbalance the main model's joint head already corrects for.
    counts = torch.bincount(flat_y, minlength=4).float().clamp_min(1)
    class_weight = (counts.sum() / (4 * counts)).clamp(0.25, 4.0)

    logger.info("Training learned fusion MLP on Dev (%d examples x %d categories = %d rows)...", n, k, flat_X.shape[0])
    fusion.train()
    for epoch in range(args.fusion_epochs):
        optimizer.zero_grad()
        logits = fusion(flat_X)
        loss = F.cross_entropy(logits, flat_y, weight=class_weight)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info("  epoch %d/%d loss=%.4f", epoch + 1, args.fusion_epochs, loss.item())

    fusion.eval()
    with torch.no_grad():
        test_pred = fusion(test_X.reshape(-1, 8)).argmax(dim=-1).reshape(n_test := test_X.shape[0], k).cpu().numpy()

    learned_metrics = acsa_f1_from_pred_joint(test_y, test_pred)
    learned_metrics["acsa_exact_match"] = float(np.mean(np.all(test_y == test_pred, axis=1)))

    print("\n=== FUSION ABLATION RESULT (same checkpoint, same Test split) ===")
    print(f"{'metric':<28}{'fixed formula (baseline)':>28}{'learned fusion (MLP)':>24}")
    for key in ("f1_micro", "f1_macro", "precision_micro", "recall_micro"):
        base_val = baseline_metrics.get(f"acsa_{key}", float("nan"))
        learn_val = learned_metrics.get(key, float("nan"))
        print(f"{key:<28}{base_val*100:>27.2f}%{learn_val*100:>23.2f}%")
    print(f"{'exact_match':<28}{baseline_metrics.get('acsa_exact_match', float('nan'))*100:>27.2f}%"
          f"{learned_metrics.get('acsa_exact_match', float('nan'))*100:>23.2f}%")
    delta = (learned_metrics["f1_micro"] - baseline_metrics["acsa_f1_micro"]) * 100
    print(f"\nDelta (learned - fixed) on acsa_f1_micro: {delta:+.2f} points")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--fusion_epochs", type=int, default=100)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
