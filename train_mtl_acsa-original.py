#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Category-Conditioned Multi-Task Learning for Vietnamese ACSA.

Architecture
------------
Review -> shared PLM encoder (default: PhoBERT-v2)
       -> category-query cross attention
       -> shared category-specific representation z_c
            |- ACD adapter/head: present vs absent
            |- Sentiment adapter/head: positive / neutral / negative
            |    with soft ACD -> sentiment gating
            `- Joint ACSA adapter/head: NONE / POS / NEU / NEG

Training supports:
- fixed loss weighting
- GradNorm dynamic loss weighting at the shared category representation z_c
- class-imbalance weights
- dev-set threshold tuning for end-to-end ACSA
- early stopping
- test metrics and JSONL predictions

Expected data format (one sample per blank-line block):
    #1
    Giá 53k size vừa.
    {DRINKS#PRICES, neutral}, {DRINKS#STYLE&OPTIONS, neutral}

Example:
    python train_mtl_acsa.py \
      --train_path Train.txt --dev_path Dev.txt --test_path Test.txt \
      --model_name vinai/phobert-base-v2 \
      --loss_weighting gradnorm \
      --output_dir outputs/phobert_mtl_acsa
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


# -----------------------------------------------------------------------------
# Labels and category descriptions
# -----------------------------------------------------------------------------
SENTIMENT2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2SENTIMENT = {v: k for k, v in SENTIMENT2ID.items()}

# Joint labels intentionally reserve 0 for NONE.
JOINT2ID = {"none": 0, "positive": 1, "neutral": 2, "negative": 3}
ID2JOINT = {v: k for k, v in JOINT2ID.items()}

CATEGORY_DESCRIPTIONS_VI = {
    "AMBIENCE#GENERAL": "không gian, bầu không khí, cách trang trí và sự thoải mái của quán",
    "DRINKS#PRICES": "giá cả và mức độ đắt rẻ của đồ uống",
    "DRINKS#QUALITY": "chất lượng, hương vị và độ ngon của đồ uống",
    "DRINKS#STYLE&OPTIONS": "loại đồ uống, cách pha chế, kích cỡ, topping và sự đa dạng lựa chọn",
    "FOOD#PRICES": "giá cả và mức độ đắt rẻ của món ăn",
    "FOOD#QUALITY": "chất lượng, hương vị, độ tươi ngon và cảm nhận về món ăn",
    "FOOD#STYLE&OPTIONS": "loại món ăn, cách chế biến, khẩu phần và sự đa dạng lựa chọn món",
    "LOCATION#GENERAL": "vị trí, địa điểm và mức độ dễ tìm của quán",
    "RESTAURANT#GENERAL": "đánh giá chung và trải nghiệm tổng thể về nhà hàng hoặc quán",
    "RESTAURANT#MISCELLANEOUS": "các tiện ích, giao hàng, giữ xe, vệ sinh, khuyến mãi và yếu tố khác của quán",
    "RESTAURANT#PRICES": "mức giá chung, hóa đơn và độ đáng tiền của nhà hàng hoặc quán",
    "SERVICE#GENERAL": "thái độ, tốc độ, sự chuyên nghiệp và chất lượng phục vụ của nhân viên",
}

LABEL_RE = re.compile(r"\{([^,{}]+),\s*([^{}]+)\}")


# -----------------------------------------------------------------------------
# Reproducibility and parsing
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Example:
    sample_id: str
    text: str
    labels: List[Tuple[str, str]]


def parse_dataset(path: str | Path) -> List[Example]:
    """Parse the uploaded ACSA format robustly, allowing multi-line text."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    examples: List[Example] = []

    for block_idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if lines[0].startswith("#"):
            sample_id = lines[0][1:].strip()
            content = lines[1:]
        else:
            sample_id = str(block_idx)
            content = lines

        ann_idx = next(
            (i for i, line in enumerate(content) if "{" in line and "}" in line),
            None,
        )
        if ann_idx is None:
            raise ValueError(
                f"No annotation line found in {path}, block {block_idx}: {block[:200]!r}"
            )

        text = " ".join(content[:ann_idx]).strip()
        ann_text = " ".join(content[ann_idx:]).strip()
        labels = [
            (cat.strip(), sent.strip().lower())
            for cat, sent in LABEL_RE.findall(ann_text)
        ]
        if not text:
            raise ValueError(f"Empty text in {path}, block {block_idx}")
        if not labels:
            raise ValueError(
                f"No valid {{CATEGORY, sentiment}} labels in {path}, block {block_idx}"
            )

        for cat, sent in labels:
            if sent not in SENTIMENT2ID:
                raise ValueError(
                    f"Unknown sentiment {sent!r} in {path}, sample #{sample_id}"
                )

        examples.append(Example(sample_id=sample_id, text=text, labels=labels))

    return examples


def infer_categories(*splits: Sequence[Example]) -> List[str]:
    cats = sorted({cat for split in splits for ex in split for cat, _ in ex.labels})
    return cats


def validate_categories(
    train: Sequence[Example], dev: Sequence[Example], test: Sequence[Example]
) -> List[str]:
    train_cats = set(infer_categories(train))
    dev_cats = set(infer_categories(dev))
    test_cats = set(infer_categories(test))
    unseen = (dev_cats | test_cats) - train_cats
    if unseen:
        raise ValueError(
            "Dev/test contain categories absent from train: " + ", ".join(sorted(unseen))
        )
    return sorted(train_cats)


def print_dataset_stats(name: str, examples: Sequence[Example]) -> None:
    cat_counts = Counter(cat for ex in examples for cat, _ in ex.labels)
    sent_counts = Counter(sent for ex in examples for _, sent in ex.labels)
    avg_labels = np.mean([len(ex.labels) for ex in examples]) if examples else 0.0
    print(f"\n[{name}] samples={len(examples):,}, avg_labels/sample={avg_labels:.3f}")
    print("  sentiment:", dict(sent_counts))
    print("  categories:")
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat:28s} {n:6d}")


# -----------------------------------------------------------------------------
# Optional Vietnamese word segmentation
# -----------------------------------------------------------------------------
def build_segmenter(name: str):
    if name == "none":
        return lambda s: s
    if name == "pyvi":
        try:
            from pyvi import ViTokenizer
        except ImportError as exc:
            raise ImportError(
                "--segmenter pyvi requires: pip install pyvi"
            ) from exc
        return ViTokenizer.tokenize
    raise ValueError(f"Unsupported segmenter: {name}")


# -----------------------------------------------------------------------------
# Dataset and collator
# -----------------------------------------------------------------------------
class ACSADataset(Dataset):
    def __init__(self, examples: Sequence[Example], categories: Sequence[str]):
        self.examples = list(examples)
        self.categories = list(categories)
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        k = len(self.categories)

        acd = np.zeros(k, dtype=np.float32)
        sent = np.full(k, -100, dtype=np.int64)  # ignored when category is absent
        joint = np.zeros(k, dtype=np.int64)      # NONE = 0

        for cat, polarity in ex.labels:
            c = self.cat2idx[cat]
            s = SENTIMENT2ID[polarity]
            acd[c] = 1.0
            sent[c] = s
            joint[c] = s + 1

        return {
            "sample_id": ex.sample_id,
            "text": ex.text,
            "acd_labels": acd,
            "sent_labels": sent,
            "joint_labels": joint,
            "gold_labels": ex.labels,
        }


def make_collate_fn(tokenizer, segmenter, max_length: int):
    def collate(batch: Sequence[Dict]) -> Dict:
        texts = [segmenter(item["text"]) for item in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            **enc,
            "sample_id": [item["sample_id"] for item in batch],
            "raw_text": [item["text"] for item in batch],
            "gold_labels": [item["gold_labels"] for item in batch],
            "acd_labels": torch.tensor(
                np.stack([item["acd_labels"] for item in batch]), dtype=torch.float32
            ),
            "sent_labels": torch.tensor(
                np.stack([item["sent_labels"] for item in batch]), dtype=torch.long
            ),
            "joint_labels": torch.tensor(
                np.stack([item["joint_labels"] for item in batch]), dtype=torch.long
            ),
        }

    return collate


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class TaskAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(F.gelu(self.down(x))))
        return self.norm(x + self.dropout(delta))


class CategoryConditionedMTL(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer,
        categories: Sequence[str],
        category_texts: Sequence[str],
        num_attention_heads: int = 8,
        adapter_dim: int = 192,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.categories = list(categories)
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.config.hidden_size
        if hidden % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden} must be divisible by num_attention_heads={num_attention_heads}"
            )

        # Category descriptions are tokenized once; they are re-encoded by the shared
        # PLM on each forward pass so gradients can update the semantic category queries.
        cat_enc = tokenizer(
            list(category_texts),
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        )
        self.register_buffer("cat_input_ids", cat_enc["input_ids"], persistent=False)
        self.register_buffer("cat_attention_mask", cat_enc["attention_mask"], persistent=False)

        self.cat_query_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )
        self.cross_ffn_norm = nn.LayerNorm(hidden)

        self.acd_adapter = TaskAdapter(hidden, adapter_dim, dropout)
        self.sent_adapter = TaskAdapter(hidden, adapter_dim, dropout)
        self.joint_adapter = TaskAdapter(hidden, adapter_dim, dropout)

        self.acd_head = nn.Linear(hidden, 1)
        self.sent_head = nn.Linear(hidden, 3)
        self.joint_head = nn.Linear(hidden, 4)

        # Learned strength for the soft ACD -> sentiment interaction.
        # sigmoid(0)=0.5 initially.
        self.raw_gate_alpha = nn.Parameter(torch.tensor(0.0))

    def _encode_category_queries(self) -> torch.Tensor:
        cat_out = self.encoder(
            input_ids=self.cat_input_ids,
            attention_mask=self.cat_attention_mask,
            return_dict=True,
        ).last_hidden_state
        # <s>/CLS-style first token representation for each semantic description.
        return self.cat_query_proj(cat_out[:, 0, :])  # [K, D]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        sent_h = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state  # [B, L, D]

        cat_q = self._encode_category_queries()  # [K, D]
        batch_size = sent_h.size(0)
        q = cat_q.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]

        attn_out, _ = self.cross_attention(
            query=q,
            key=sent_h,
            value=sent_h,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        z = self.cross_norm(q + attn_out)
        z = self.cross_ffn_norm(z + self.cross_ffn(z))  # shared branching representation

        acd_z = self.acd_adapter(z)
        acd_logits = self.acd_head(acd_z).squeeze(-1)  # [B, K]

        # Soft gate: sentiment remains trainable even when ACD is uncertain/wrong.
        acd_prob = torch.sigmoid(acd_logits)
        alpha = torch.sigmoid(self.raw_gate_alpha)
        sent_input = z * (1.0 + alpha * acd_prob.unsqueeze(-1))
        sent_z = self.sent_adapter(sent_input)
        sent_logits = self.sent_head(sent_z)  # [B, K, 3]

        joint_z = self.joint_adapter(z)
        joint_logits = self.joint_head(joint_z)  # [B, K, 4]

        return {
            "acd_logits": acd_logits,
            "sent_logits": sent_logits,
            "joint_logits": joint_logits,
            "shared_z": z,
            "gate_alpha": alpha,
        }


# -----------------------------------------------------------------------------
# Imbalance-aware losses
# -----------------------------------------------------------------------------
@dataclass
class LossWeights:
    acd_pos_weight: torch.Tensor
    sent_class_weight: torch.Tensor
    joint_class_weight: torch.Tensor


def _balanced_weights(counts: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency weights softened by sqrt (power=0.5)."""
    counts = counts.astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (len(counts) * counts)
    weights = inv ** power
    weights = weights / weights.mean()
    return np.clip(weights, 0.25, 4.0).astype(np.float32)


def compute_loss_weights(dataset: ACSADataset) -> LossWeights:
    acd_rows, sent_rows, joint_rows = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        acd_rows.append(item["acd_labels"])
        sent_rows.append(item["sent_labels"])
        joint_rows.append(item["joint_labels"])

    acd = np.stack(acd_rows)
    sent = np.stack(sent_rows)
    joint = np.stack(joint_rows)

    pos = acd.sum(axis=0)
    neg = len(dataset) - pos
    # sqrt dampens very large rare-category positive weights.
    acd_pos_weight = np.sqrt((neg + 1.0) / (pos + 1.0)).astype(np.float32)
    acd_pos_weight = np.clip(acd_pos_weight, 1.0, 6.0)

    sent_present = sent[sent != -100]
    sent_counts = np.bincount(sent_present, minlength=3)
    joint_counts = np.bincount(joint.reshape(-1), minlength=4)

    return LossWeights(
        acd_pos_weight=torch.tensor(acd_pos_weight),
        sent_class_weight=torch.tensor(_balanced_weights(sent_counts)),
        joint_class_weight=torch.tensor(_balanced_weights(joint_counts)),
    )


def compute_task_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = outputs["acd_logits"].device
    acd_loss = F.binary_cross_entropy_with_logits(
        outputs["acd_logits"],
        batch["acd_labels"],
        pos_weight=weights.acd_pos_weight.to(device),
    )
    sent_loss = F.cross_entropy(
        outputs["sent_logits"].reshape(-1, 3),
        batch["sent_labels"].reshape(-1),
        weight=weights.sent_class_weight.to(device),
        ignore_index=-100,
    )
    joint_loss = F.cross_entropy(
        outputs["joint_logits"].reshape(-1, 4),
        batch["joint_labels"].reshape(-1),
        weight=weights.joint_class_weight.to(device),
    )
    return acd_loss, sent_loss, joint_loss


# -----------------------------------------------------------------------------
# GradNorm
# -----------------------------------------------------------------------------
class GradNormBalancer(nn.Module):
    """
    Dynamic task weighting based on GradNorm.

    Gradient norms are measured at `shared_z`, the representation immediately
    before the three task-specific adapters. This makes the method efficient and
    directly measures conflict/imbalance at the task branching point.
    """

    def __init__(self, num_tasks: int = 3, alpha: float = 1.5):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(num_tasks))
        self.alpha = alpha
        self.register_buffer("initial_losses", torch.zeros(num_tasks))
        self.initialized = False

    def normalized_weights(self) -> torch.Tensor:
        w = F.softplus(self.raw_weights) + 1e-6
        return len(w) * w / w.sum()

    def compute_weight_gradient(
        self,
        losses: Sequence[torch.Tensor],
        shared_representation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_vec = torch.stack(list(losses))
        if not self.initialized:
            self.initial_losses.copy_(loss_vec.detach().clamp_min(1e-8))
            self.initialized = True

        w = self.normalized_weights()
        grad_norms = []
        for i, loss in enumerate(losses):
            grad = torch.autograd.grad(
                w[i] * loss,
                shared_representation,
                retain_graph=True,
                create_graph=True,
            )[0]
            grad_norms.append(torch.norm(grad, p=2))
        grad_norms = torch.stack(grad_norms)

        with torch.no_grad():
            loss_ratio = loss_vec.detach() / self.initial_losses.clamp_min(1e-8)
            inverse_train_rate = loss_ratio / loss_ratio.mean()
            target = grad_norms.detach().mean() * (inverse_train_rate ** self.alpha)

        gradnorm_objective = torch.abs(grad_norms - target).sum()
        weight_grad = torch.autograd.grad(
            gradnorm_objective,
            self.raw_weights,
            retain_graph=True,
            create_graph=False,
        )[0]
        return w, weight_grad, gradnorm_objective.detach()


# -----------------------------------------------------------------------------
# Prediction fusion and metrics
# -----------------------------------------------------------------------------
def fuse_predictions(
    acd_logits: np.ndarray,
    sent_logits: np.ndarray,
    joint_logits: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse ACD + sentiment + joint heads for end-to-end inference.

    Returns
    -------
    pred_joint: [N, K], 0=NONE, 1=POS, 2=NEU, 3=NEG
    presence_score: [N, K]
    """
    acd_prob = 1.0 / (1.0 + np.exp(-np.clip(acd_logits, -30, 30)))

    joint_shift = joint_logits - joint_logits.max(axis=-1, keepdims=True)
    joint_prob = np.exp(joint_shift)
    joint_prob /= joint_prob.sum(axis=-1, keepdims=True)
    joint_presence = 1.0 - joint_prob[..., 0]

    sent_shift = sent_logits - sent_logits.max(axis=-1, keepdims=True)
    sent_prob = np.exp(sent_shift)
    sent_prob /= sent_prob.sum(axis=-1, keepdims=True)

    joint_sent = joint_prob[..., 1:]
    joint_sent /= np.clip(joint_sent.sum(axis=-1, keepdims=True), 1e-8, None)

    presence_score = 0.5 * acd_prob + 0.5 * joint_presence
    sentiment_score = 0.5 * sent_prob + 0.5 * joint_sent
    sentiment_id = sentiment_score.argmax(axis=-1) + 1
    pred_joint = np.where(presence_score >= threshold, sentiment_id, 0)
    return pred_joint.astype(np.int64), presence_score


def build_multilabel_pair_matrix(joint: np.ndarray) -> np.ndarray:
    """Convert [N,K] joint labels into [N, K*3] category-polarity indicators."""
    n, k = joint.shape
    y = np.zeros((n, k * 3), dtype=np.int64)
    for s in range(1, 4):
        mask = joint == s
        rows, cols = np.where(mask)
        y[rows, cols * 3 + (s - 1)] = 1
    return y


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> Dict[str, float]:
    out = {}
    for avg in ("micro", "macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=avg,
            zero_division=0,
        )
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f1)
    return out


def compute_metrics(
    raw: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    gold_acd = raw["acd_labels"].astype(np.int64)
    gold_sent = raw["sent_labels"].astype(np.int64)
    gold_joint = raw["joint_labels"].astype(np.int64)

    pred_joint, presence_score = fuse_predictions(
        raw["acd_logits"], raw["sent_logits"], raw["joint_logits"], threshold
    )
    pred_acd = (presence_score >= threshold).astype(np.int64)

    # ACD: flatten all sample-category decisions.
    acd_metrics = classification_metrics(gold_acd.reshape(-1), pred_acd.reshape(-1), labels=[0, 1])

    # Oracle sentiment: evaluate sentiment head only where the gold category is present.
    sent_pred = raw["sent_logits"].argmax(axis=-1)
    present_mask = gold_sent != -100
    if present_mask.any():
        sent_metrics = classification_metrics(
            gold_sent[present_mask], sent_pred[present_mask], labels=[0, 1, 2]
        )
    else:
        sent_metrics = {k: 0.0 for k in classification_metrics(np.array([0]), np.array([0]), labels=[0,1,2])}

    # End-to-end ACSA: each category-polarity pair is a multilabel target.
    gold_pair = build_multilabel_pair_matrix(gold_joint)
    pred_pair = build_multilabel_pair_matrix(pred_joint)
    pair_metrics = classification_metrics(gold_pair, pred_pair)

    # Exact sample match is strict: all 12 categories + sentiments must match.
    exact_match = float(np.mean(np.all(gold_joint == pred_joint, axis=1)))

    # Category detection subset accuracy.
    acd_exact = float(np.mean(np.all((gold_joint > 0) == (pred_joint > 0), axis=1)))

    metrics = {f"acd_{k}": v for k, v in acd_metrics.items()}
    metrics.update({f"sent_oracle_{k}": v for k, v in sent_metrics.items()})
    metrics.update({f"acsa_{k}": v for k, v in pair_metrics.items()})
    metrics["acsa_exact_match"] = exact_match
    metrics["acd_exact_match"] = acd_exact
    metrics["threshold"] = float(threshold)
    return metrics, pred_joint


def tune_threshold(raw: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float], np.ndarray]:
    best = (-1.0, 0.5, None, None)
    for threshold in np.arange(0.20, 0.81, 0.02):
        metrics, pred = compute_metrics(raw, float(threshold))
        score = metrics["acsa_f1_micro"]
        if score > best[0]:
            best = (score, float(threshold), metrics, pred)
    _, threshold, metrics, pred = best
    return threshold, metrics, pred


# -----------------------------------------------------------------------------
# Train / eval utilities
# -----------------------------------------------------------------------------
def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = dict(batch)
    for key in ("input_ids", "attention_mask", "token_type_ids", "acd_labels", "sent_labels", "joint_labels"):
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device, non_blocking=True)
    return out


@torch.no_grad()
def collect_outputs(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    storage = {
        "acd_logits": [],
        "sent_logits": [],
        "joint_logits": [],
        "acd_labels": [],
        "sent_labels": [],
        "joint_labels": [],
        "sample_id": [],
        "raw_text": [],
        "gold_labels": [],
    }

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        for key in ("acd_logits", "sent_logits", "joint_logits"):
            storage[key].append(outputs[key].detach().cpu().float().numpy())
        for key in ("acd_labels", "sent_labels", "joint_labels"):
            storage[key].append(batch[key].detach().cpu().numpy())
        storage["sample_id"].extend(batch["sample_id"])
        storage["raw_text"].extend(batch["raw_text"])
        storage["gold_labels"].extend(batch["gold_labels"])

    for key in ("acd_logits", "sent_logits", "joint_logits", "acd_labels", "sent_labels", "joint_labels"):
        storage[key] = np.concatenate(storage[key], axis=0)
    return storage


def build_optimizer(model: nn.Module, encoder_lr: float, head_lr: float, weight_decay: float):
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    return AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    categories: Sequence[str],
    threshold: float,
    dev_metrics: Dict[str, float],
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "categories": list(categories),
            "threshold": threshold,
            "dev_metrics": dev_metrics,
            "args": vars(args),
        },
        path,
    )


def write_predictions(
    path: Path,
    raw: Dict[str, np.ndarray],
    pred_joint: np.ndarray,
    categories: Sequence[str],
):
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(pred_joint)):
            pred_labels = []
            for c, label_id in enumerate(pred_joint[i]):
                if label_id == 0:
                    continue
                pred_labels.append(
                    {
                        "category": categories[c],
                        "sentiment": ID2JOINT[int(label_id)],
                    }
                )
            record = {
                "id": raw["sample_id"][i],
                "text": raw["raw_text"][i],
                "gold": [
                    {"category": c, "sentiment": s} for c, s in raw["gold_labels"][i]
                ],
                "prediction": pred_labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = parse_dataset(args.train_path)
    dev_examples = parse_dataset(args.dev_path)
    test_examples = parse_dataset(args.test_path)
    categories = validate_categories(train_examples, dev_examples, test_examples)

    print_dataset_stats("train", train_examples)
    print_dataset_stats("dev", dev_examples)
    print_dataset_stats("test", test_examples)
    print("\nCategories:", categories)

    segmenter = build_segmenter(args.segmenter)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = ACSADataset(train_examples, categories)
    dev_ds = ACSADataset(dev_examples, categories)
    test_ds = ACSADataset(test_examples, categories)

    collate = make_collate_fn(tokenizer, segmenter, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    category_texts = []
    for cat in categories:
        desc = CATEGORY_DESCRIPTIONS_VI.get(cat, cat.replace("#", " ").replace("&", " và "))
        category_texts.append(segmenter(desc))

    model = CategoryConditionedMTL(
        model_name=args.model_name,
        tokenizer=tokenizer,
        categories=categories,
        category_texts=category_texts,
        num_attention_heads=args.num_attention_heads,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    print(f"\nDevice: {device}")

    loss_weights = compute_loss_weights(train_ds)
    print("ACD pos weights:", loss_weights.acd_pos_weight.tolist())
    print("Sentiment class weights [pos, neu, neg]:", loss_weights.sent_class_weight.tolist())
    print("Joint class weights [none, pos, neu, neg]:", loss_weights.joint_class_weight.tolist())

    optimizer = build_optimizer(model, args.encoder_lr, args.head_lr, args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    gradnorm = None
    gradnorm_optimizer = None
    if args.loss_weighting == "gradnorm":
        gradnorm = GradNormBalancer(num_tasks=3, alpha=args.gradnorm_alpha).to(device)
        gradnorm_optimizer = AdamW([gradnorm.raw_weights], lr=args.gradnorm_lr, weight_decay=0.0)

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.5
    patience_counter = 0
    checkpoint_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = np.zeros(4, dtype=np.float64)  # total, acd, sent, joint
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")

        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if gradnorm_optimizer is not None:
                gradnorm_optimizer.zero_grad(set_to_none=True)

            outputs = model(batch["input_ids"], batch["attention_mask"])
            task_losses = compute_task_losses(outputs, batch, loss_weights)

            if args.loss_weighting == "gradnorm":
                assert gradnorm is not None and gradnorm_optimizer is not None
                task_w, grad_w, gradnorm_obj = gradnorm.compute_weight_gradient(
                    task_losses, outputs["shared_z"]
                )
                # Network parameters are optimized with current task weights treated as constants.
                total_loss = sum(w.detach() * loss for w, loss in zip(task_w, task_losses))
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # The GradNorm objective updates only task weights.
                gradnorm.raw_weights.grad = grad_w.detach()
                gradnorm_optimizer.step()
                display_w = gradnorm.normalized_weights().detach().cpu().tolist()
            else:
                fixed = torch.tensor(
                    [args.lambda_acd, args.lambda_sent, args.lambda_joint],
                    dtype=task_losses[0].dtype,
                    device=device,
                )
                total_loss = sum(w * loss for w, loss in zip(fixed, task_losses))
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                display_w = fixed.detach().cpu().tolist()

            vals = [total_loss.item()] + [x.item() for x in task_losses]
            running += np.asarray(vals)
            pbar.set_postfix(
                loss=f"{running[0]/step:.4f}",
                acd=f"{running[1]/step:.3f}",
                sent=f"{running[2]/step:.3f}",
                joint=f"{running[3]/step:.3f}",
                w="/".join(f"{x:.2f}" for x in display_w),
                gate=f"{outputs['gate_alpha'].item():.2f}",
            )

        # Tune the ACSA presence threshold on dev for each epoch.
        dev_raw = collect_outputs(model, dev_loader, device)
        threshold, dev_metrics, _ = tune_threshold(dev_raw)
        score = dev_metrics["acsa_f1_micro"]

        epoch_record = {
            "epoch": epoch,
            "train_total_loss": float(running[0] / max(1, len(train_loader))),
            "train_acd_loss": float(running[1] / max(1, len(train_loader))),
            "train_sent_loss": float(running[2] / max(1, len(train_loader))),
            "train_joint_loss": float(running[3] / max(1, len(train_loader))),
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}: dev ACSA micro-F1={score:.4f}, "
            f"macro-F1={dev_metrics['acsa_f1_macro']:.4f}, "
            f"threshold={threshold:.2f}, exact={dev_metrics['acsa_exact_match']:.4f}"
        )

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            patience_counter = 0
            save_checkpoint(
                checkpoint_path, model, args, categories, threshold, dev_metrics
            )
            print(f"  -> saved new best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch={best_epoch}")
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Final evaluation with the best dev checkpoint and its tuned threshold.
    # ------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = float(checkpoint["threshold"])
    model.eval()

    dev_raw = collect_outputs(model, dev_loader, device)
    dev_metrics, dev_pred = compute_metrics(dev_raw, best_threshold)
    test_raw = collect_outputs(model, test_loader, device)
    test_metrics, test_pred = compute_metrics(test_raw, best_threshold)

    results = {
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "categories": categories,
        "dev": dev_metrics,
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_predictions(output_dir / "dev_predictions.jsonl", dev_raw, dev_pred, categories)
    write_predictions(output_dir / "test_predictions.jsonl", test_raw, test_pred, categories)

    print("\n=== BEST MODEL ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Dev-tuned threshold: {best_threshold:.2f}")
    print("\n=== TEST ===")
    for key, value in test_metrics.items():
        print(f"{key:32s}: {value:.6f}")
    print(f"\nArtifacts written to: {output_dir.resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Category-conditioned multi-task ACSA trainer")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--dev_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/mtl_acsa")

    p.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    p.add_argument("--segmenter", choices=["none", "pyvi"], default="pyvi")
    p.add_argument("--max_length", type=int, default=160)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--adapter_dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)

    p.add_argument("--loss_weighting", choices=["fixed", "gradnorm"], default="gradnorm")
    p.add_argument("--lambda_acd", type=float, default=1.0)
    p.add_argument("--lambda_sent", type=float, default=1.0)
    p.add_argument("--lambda_joint", type=float, default=0.5)
    p.add_argument("--gradnorm_alpha", type=float, default=1.5)
    p.add_argument("--gradnorm_lr", type=float, default=2.5e-3)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
