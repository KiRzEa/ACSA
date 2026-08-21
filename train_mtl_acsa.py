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
import logging
import math
import os
import random
import re
import sys
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

import mapper

logger = logging.getLogger("mtl_acsa")


def configure_logging(output_dir: Path) -> None:
    """Console + output_dir/train.log, so an anomaly is visible live and still
    recoverable after a Colab/Kaggle session disconnects."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


# -----------------------------------------------------------------------------
# Labels, domains and category descriptions
# -----------------------------------------------------------------------------
SENTIMENT2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2SENTIMENT = {v: k for k, v in SENTIMENT2ID.items()}

# Joint labels intentionally reserve 0 for NONE.
JOINT2ID = {"none": 0, "positive": 1, "neutral": 2, "negative": 3}
ID2JOINT = {v: k for k, v in JOINT2ID.items()}

# Kept for backward compatibility with callers that imported this name directly;
# it's also the fallback description set when --domain is not given.
CATEGORY_DESCRIPTIONS_VI = mapper.CATEGORY_DESCRIPTIONS["Restaurant"]

# --domain CLI value -> mapper.py / ABSA_LLMs Pair-data domain key.
DOMAIN_KEY = {
    "restaurant": "Restaurant",
    "restaurant_clean": "Restaurant",
    "hotel": "Hotel",
    "hotel_clean": "Hotel",
    "phone": "Phone",
    "education": "Education",
    "beauty": "Beauty",
    "mother": "Mother",
    "technology": "Technology",
}
# Domains with their own original raw-text *_ABSA/*.txt files, whose category
# codes already match mapper.py's *_dict keys exactly (no translation needed,
# unlike e.g. Education). "restaurant" and "hotel" train from these directly
# (raw text; "restaurant" is the validated 77.43 F1 baseline). Every other
# domain, including the "_clean" variants, sources the exact same underlying
# samples but from ABSA_LLMs' cleaned data/Pair/<Domain> CSVs (see
# mapper.load_pair_examples) -- "restaurant_clean"/"hotel_clean" exist to A/B
# clean_doc() preprocessing against the raw-text version of the same domain.
#
# NOTE: verified Hotel_ABSA/*.txt and data/Pair/Hotel/*.csv are the same
# underlying ~10k-example pool (label distributions match almost exactly), but
# NOT the same train/dev/test split -- ABSA_LLMs re-shuffled it, so "hotel" and
# "hotel_clean" are not evaluated on identical held-out sets. Restaurant's two
# variants ARE sample-for-sample identical (verified via raw_input/raw_output).
RAW_TEXT_DOMAIN_ROOTS = {
    "restaurant": "Res_ABSA",
    "hotel": "Hotel_ABSA",
}
ABSA_LLMS_DOMAINS = set(DOMAIN_KEY) - set(RAW_TEXT_DOMAIN_ROOTS)

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def load_pair_split(domain_key: str, csv_path: str | Path) -> List[Example]:
    """Load one ABSA_LLMs data/Pair/<Domain>/{Train,Dev,Test}.csv split into
    Examples, via mapper.load_pair_examples (see mapper.py for the label
    recovery logic for domains that only ship a flattened NL 'output' column)."""
    tuples, skipped = mapper.load_pair_examples(domain_key, str(csv_path))
    if skipped:
        logger.warning("skipped %d unparseable/empty rows in %s", skipped, csv_path)
    return [Example(sample_id=sid, text=text, labels=labels) for sid, text, labels in tuples]


def print_dataset_stats(name: str, examples: Sequence[Example]) -> None:
    cat_counts = Counter(cat for ex in examples for cat, _ in ex.labels)
    sent_counts = Counter(sent for ex in examples for _, sent in ex.labels)
    avg_labels = np.mean([len(ex.labels) for ex in examples]) if examples else 0.0
    logger.info("[%s] samples=%d, avg_labels/sample=%.3f", name, len(examples), avg_labels)
    logger.info("  sentiment: %s", dict(sent_counts))
    if examples and min(cat_counts.values()) < 5:
        rare = {c: n for c, n in cat_counts.items() if n < 5}
        logger.warning("  categories with <5 examples in [%s] (unreliable metrics): %s", name, rare)
    logger.info("  categories:")
    for cat, n in sorted(cat_counts.items()):
        logger.info("    %-28s %6d", cat, n)


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
        category_self_attention: bool = False,
    ):
        super().__init__()
        self.category_self_attention_enabled = category_self_attention
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

        # Optional: lets each category's representation attend to every other
        # category's representation within the same review, after the text
        # cross-attention. Categories aren't independent (e.g. ROOMS#CLEANLINESS
        # and ROOM_AMENITIES#CLEANLINESS correlate) -- this lets rare categories
        # borrow statistical signal from correlated, well-observed ones,
        # dynamically per sample. A targeted inductive bias for the rare-category
        # problem, not generic added capacity (unlike the num_attention_heads/
        # adapter_dim sweep, which just made the same per-category computation
        # wider and empirically made things worse).
        if category_self_attention:
            self.category_self_attn = nn.MultiheadAttention(
                embed_dim=hidden,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.category_self_attn_norm = nn.LayerNorm(hidden)

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

        if self.category_self_attention_enabled:
            # No key_padding_mask needed: all K category positions are always
            # valid (unlike the token sequence, categories are never padded).
            self_attn_out, _ = self.category_self_attn(query=z, key=z, value=z, need_weights=False)
            z = self.category_self_attn_norm(z + self_attn_out)

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

    clipped = [dataset.categories[i] for i in np.where(acd_pos_weight >= 6.0)[0]]
    if clipped:
        logger.warning(
            "%d categories hit the ACD positive-weight clip ceiling (severe imbalance, <~1/36 positive rate): %s",
            len(clipped), clipped,
        )

    return LossWeights(
        acd_pos_weight=torch.tensor(acd_pos_weight),
        sent_class_weight=torch.tensor(_balanced_weights(sent_counts)),
        joint_class_weight=torch.tensor(_balanced_weights(joint_counts)),
    )


def compute_oversample_weights(dataset: ACSADataset, power: float = 0.5) -> torch.Tensor:
    """
    Per-example sampling weight for a WeightedRandomSampler: examples touching
    at least one rare category get sampled more often per epoch, so the model
    sees rare categories more without needing new data. Unlike the loss-level
    weights above (which reweight the *gradient* of examples the model already
    sees equally often), this changes how often each example is *drawn* in the
    first place -- a different lever, since none of the loss-side techniques
    tried moved the aggregate ACSA F1 much.

    Weight for example i = max over its present categories c of
    (N / count[c]) ** power, then floored at 1.0 (most common category's
    score) so this only ever *boosts* rare-category examples, never
    downweights common ones. Won't meaningfully help categories with only a
    handful of examples total (e.g. 2) -- oversampling just repeats the same
    few examples, risking memorization of their exact phrasing rather than
    learning anything generalizable; that's a real data-scarcity ceiling, not
    something this fixes.
    """
    n = len(dataset)
    acd_rows = [dataset[i]["acd_labels"] for i in range(n)]
    acd = np.stack(acd_rows)  # [N, K]

    counts = np.maximum(acd.sum(axis=0), 1.0)  # [K]
    cat_score = (n / counts) ** power
    cat_score = cat_score / cat_score.min()  # floor at 1.0 for the most common category

    weights = np.ones(n, dtype=np.float64)
    for i in range(n):
        present = np.where(acd[i] == 1)[0]
        if len(present) > 0:
            weights[i] = cat_score[present].max()

    logger.info(
        "Oversample weights: min=%.2f max=%.2f mean=%.2f (most-boosted example weight -> rarest "
        "category it contains gets sampled ~%.1fx as often as the most common category)",
        weights.min(), weights.max(), weights.mean(), weights.max(),
    )
    return torch.tensor(weights, dtype=torch.double)


@dataclass
class LossConfig:
    acd_loss_fn: str = "bce"      # "bce", "focal", or "asl"
    sent_loss_fn: str = "ce"      # "ce" or "focal"
    joint_loss_fn: str = "ce"     # "ce" or "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0  # only applied to "ce" variants
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05


def binary_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float, pos_weight: torch.Tensor
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for the multi-label ACD head: down-weights
    already-easy (high-confidence-correct) category decisions so rare/hard
    categories contribute proportionally more to the gradient than plain BCE
    with only inverse-frequency pos_weight gives them."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1.0 - prob)
    return ((1.0 - pt).clamp_min(0.0) ** gamma * bce).mean()


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Asymmetric Loss (Ben-Baruch et al. 2021, https://arxiv.org/abs/2009.14119)
    for the multi-label ACD head. Unlike focal loss, which treats positive and
    negative labels the same, ASL treats them asymmetrically -- appropriate
    here because across 34 categories most (category, example) pairs are
    negative (absent). Two mechanisms, both applied only to negatives:
      - Probability shifting: shifts each negative's "absent" confidence up by
        `clip` before computing its loss, so an already-easy negative (e.g.
        absent-confidence 0.97) gets pushed to ~1.0 and contributes exactly
        zero loss -- hard-discarding trivially-easy negatives rather than just
        down-weighting them.
      - A steeper focusing exponent on negatives (gamma_neg) than positives
        (gamma_pos), so remaining negatives are still down-weighted more
        aggressively than positives once shifting is applied.
    Positives keep gamma_pos=0 (no down-weighting) by default: they're the
    scarce, valuable signal here, unlike negatives which vastly outnumber them.
    """
    prob = torch.sigmoid(logits)
    prob_pos = prob
    prob_neg = 1.0 - prob
    if clip > 0:
        prob_neg = (prob_neg + clip).clamp(max=1.0)

    loss_pos = targets * torch.log(prob_pos.clamp(min=eps))
    loss_neg = (1.0 - targets) * torch.log(prob_neg.clamp(min=eps))
    loss = loss_pos + loss_neg

    if gamma_pos > 0 or gamma_neg > 0:
        with torch.no_grad():
            pt = prob_pos * targets + prob_neg * (1.0 - targets)
            focal_weight = (1.0 - pt) ** (gamma_pos * targets + gamma_neg * (1.0 - targets))
        loss = loss * focal_weight

    return -loss.mean()


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Focal loss for the sentiment/joint heads. ignore_index rows (absent
    categories in the sentiment head) contribute 0 loss, matching F.cross_entropy's
    ignore_index behavior, and are excluded from the mean.

    pt (the true predicted probability of the correct class) is derived from
    the *unweighted* cross-entropy: exp(-ce) only equals pt when weight=None,
    since a weighted ce is -w_c * log(pt), not -log(pt). Deriving pt from the
    weighted loss would silently distort the focal term by p^w_c instead of p.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index, reduction="none")
    with torch.no_grad():
        if weight is None:
            pt = torch.exp(-ce)
        else:
            ce_unweighted = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")
            pt = torch.exp(-ce_unweighted)
    focal = (1.0 - pt).clamp_min(0.0) ** gamma * ce
    mask = (targets != ignore_index).float()
    return (focal * mask).sum() / mask.sum().clamp_min(1.0)


def compute_task_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
    loss_config: LossConfig = LossConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = outputs["acd_logits"].device
    acd_pos_weight = weights.acd_pos_weight.to(device)
    sent_weight = weights.sent_class_weight.to(device)
    joint_weight = weights.joint_class_weight.to(device)

    if loss_config.acd_loss_fn == "focal":
        acd_loss = binary_focal_loss(
            outputs["acd_logits"], batch["acd_labels"], loss_config.focal_gamma, acd_pos_weight
        )
    elif loss_config.acd_loss_fn == "asl":
        # ASL handles imbalance itself via asymmetric shifting/focusing -- deliberately
        # not combined with acd_pos_weight, to avoid double-correcting the same imbalance
        # the way stacking focal loss on top of inverse-frequency weights did previously.
        acd_loss = asymmetric_loss(
            outputs["acd_logits"], batch["acd_labels"],
            gamma_pos=loss_config.asl_gamma_pos, gamma_neg=loss_config.asl_gamma_neg,
            clip=loss_config.asl_clip,
        )
    else:
        acd_loss = F.binary_cross_entropy_with_logits(
            outputs["acd_logits"], batch["acd_labels"], pos_weight=acd_pos_weight
        )

    if loss_config.sent_loss_fn == "focal":
        # Not combined with sent_class_weight: focal's (1-pt)^gamma already
        # up-weights rare/hard classes, same as class weights do -- stacking
        # both double-corrects the same imbalance (mirrors the acd_pos_weight
        # exclusion for ACD's "asl" branch above). Observed empirically: runs
        # using focal on sent/joint had abnormally large early-epoch gradient
        # norms (~100-180 vs ~20-50 for bce/asl runs) and underperformed every
        # bce baseline, consistent with over-amplified gradients on already-
        # up-weighted rare classes.
        sent_loss = multiclass_focal_loss(
            outputs["sent_logits"].reshape(-1, 3),
            batch["sent_labels"].reshape(-1),
            loss_config.focal_gamma,
            weight=None,
            ignore_index=-100,
        )
    else:
        sent_loss = F.cross_entropy(
            outputs["sent_logits"].reshape(-1, 3),
            batch["sent_labels"].reshape(-1),
            weight=sent_weight,
            ignore_index=-100,
            label_smoothing=loss_config.label_smoothing,
        )

    if loss_config.joint_loss_fn == "focal":
        # See sent_loss_fn note above: not combined with joint_class_weight.
        joint_loss = multiclass_focal_loss(
            outputs["joint_logits"].reshape(-1, 4),
            batch["joint_labels"].reshape(-1),
            loss_config.focal_gamma,
            weight=None,
            ignore_index=-100,  # never actually ignored here (joint has no -100 labels)
        )
    else:
        joint_loss = F.cross_entropy(
            outputs["joint_logits"].reshape(-1, 4),
            batch["joint_labels"].reshape(-1),
            weight=joint_weight,
            label_smoothing=loss_config.label_smoothing,
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
    # threshold may be a scalar (global) or a per-category array; store a
    # JSON-safe scalar summary here regardless, the full array (when used) is
    # saved separately with the checkpoint/results.
    metrics["threshold"] = float(np.mean(threshold))
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


def tune_thresholds_per_category(
    raw: Dict[str, np.ndarray], num_categories: int, min_support: int = 20
) -> np.ndarray:
    """Independently pick the presence threshold that maximizes each category's
    own (category, polarity) micro-F1, instead of one threshold shared across
    all categories. Rare categories often need a different operating point than
    frequent ones -- a single global threshold under-serves them. Pure
    post-hoc reranking of already-collected logits, no retraining involved.

    Categories with fewer than min_support gold-positive dev examples skip the
    per-category sweep and keep the single global threshold instead: picking a
    threshold from e.g. 2-7 examples is closer to fitting noise than tuning
    (observed directly -- Hotel's FACILITIES#PRICES has 2 dev examples and its
    per-category threshold swung precision/recall wildly relative to the
    global-threshold baseline on test).
    """
    gold_joint = raw["joint_labels"].astype(np.int64)
    gold_pair = build_multilabel_pair_matrix(gold_joint)
    support = (gold_joint > 0).sum(axis=0)  # gold-positive dev count per category

    global_threshold, _, _ = tune_threshold(raw)
    global_pred_joint, _ = fuse_predictions(
        raw["acd_logits"], raw["sent_logits"], raw["joint_logits"], global_threshold
    )
    global_pred_pair = build_multilabel_pair_matrix(global_pred_joint)

    best_threshold = np.full(num_categories, global_threshold, dtype=np.float64)
    best_f1 = np.zeros(num_categories, dtype=np.float64)
    for c in range(num_categories):
        cols = slice(c * 3, c * 3 + 3)
        _, _, f1, _ = precision_recall_fscore_support(
            gold_pair[:, cols], global_pred_pair[:, cols], average="micro", zero_division=0
        )
        best_f1[c] = f1

    for threshold in np.arange(0.20, 0.81, 0.02):
        pred_joint, _ = fuse_predictions(
            raw["acd_logits"], raw["sent_logits"], raw["joint_logits"], float(threshold)
        )
        pred_pair = build_multilabel_pair_matrix(pred_joint)
        for c in range(num_categories):
            if support[c] < min_support:
                continue
            cols = slice(c * 3, c * 3 + 3)
            _, _, f1, _ = precision_recall_fscore_support(
                gold_pair[:, cols], pred_pair[:, cols], average="micro", zero_division=0
            )
            if f1 > best_f1[c]:
                best_f1[c] = f1
                best_threshold[c] = float(threshold)

    return best_threshold


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


_ENCODER_LAYER_RE = re.compile(r"^encoder\.encoder\.layer\.(\d+)\.")


def _encoder_layer_depth(name: str, num_layers: int) -> int:
    """0 = embeddings (bottom), num_layers+1 = anything past the last
    transformer layer (e.g. a pooler), matching standard LLRD conventions."""
    if name.startswith("encoder.embeddings"):
        return 0
    match = _ENCODER_LAYER_RE.match(name)
    if match:
        return int(match.group(1)) + 1
    return num_layers + 1


def build_optimizer(
    model: nn.Module,
    encoder_lr: float,
    head_lr: float,
    gate_lr: Optional[float],
    weight_decay: float,
    llrd_decay: float = 1.0,
):
    """
    Three logical parameter groups:
      - encoder.*: base encoder_lr, optionally decayed per-layer toward the
        embeddings (llrd_decay < 1.0 enables layer-wise LR decay -- the top
        transformer layer gets the full encoder_lr, lower layers get
        progressively less, a standard trick for fine-tuning pretrained
        encoders). llrd_decay=1.0 (default) means every layer gets encoder_lr,
        i.e. LLRD is off.
      - raw_gate_alpha: if gate_lr is given (not None), its own gate_lr and
        zero weight_decay. As a single scalar with a very diffuse gradient
        path (shared across every category/sample), it needs a much higher LR
        than the rest of the heads to move at all in a few epochs -- and
        weight_decay actively pulls it back toward 0 (== alpha=0.5, inactive
        gating), which was likely fighting any real movement when it shared
        head_lr/weight_decay with everything else. gate_lr=None (default)
        instead lumps it into head_lr/weight_decay like everything else below
        -- matches the original script's behavior exactly for anyone not
        explicitly opting into the fix via --gate_lr.
      - everything else (adapters, heads, cross-attention, GradNorm excluded
        -- that has its own optimizer, and raw_gate_alpha when gate_lr=None):
        head_lr.
    """
    num_layers = model.encoder.config.num_hidden_layers
    groups: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "raw_gate_alpha" and gate_lr is not None:
            lr, wd = gate_lr, 0.0
        elif name.startswith("encoder."):
            depth = _encoder_layer_depth(name, num_layers)
            lr = encoder_lr * (llrd_decay ** (num_layers + 1 - depth))
            wd = weight_decay
        else:
            lr, wd = head_lr, weight_decay
        groups.setdefault((lr, wd), []).append(param)

    return AdamW(
        [{"params": params, "lr": lr, "weight_decay": wd} for (lr, wd), params in groups.items()]
    )


# -----------------------------------------------------------------------------
# Child-Tuning (Xu et al., "Raise a Child in Large Language Model", EMNLP 2021)
# -----------------------------------------------------------------------------
def estimate_child_tuning_mask(
    model: nn.Module,
    loader: DataLoader,
    loss_weights: LossWeights,
    loss_config: LossConfig,
    device: torch.device,
    num_batches: int,
    ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Child-Tuning_D: estimate each encoder parameter's Fisher information (mean
    squared gradient over a calibration pass) on the pretrained model, before
    any fine-tuning happens, and keep only the top `ratio` fraction as the
    "child network" -- the rest get their gradients zeroed every step during
    training. Only applied to encoder.* parameters: the heads/adapters/
    cross-attention are freshly initialized and need full gradient flow to
    learn from scratch, whereas the whole point of Child-Tuning is protecting
    *pretrained* knowledge in the encoder from overfitting on a small dataset.
    """
    was_training = model.training
    model.eval()  # calibration in eval mode: a stable estimate, not noised by dropout

    fisher = {
        name: torch.zeros_like(p)
        for name, p in model.named_parameters()
        if name.startswith("encoder.") and p.requires_grad
    }

    n_done = 0
    for batch in loader:
        if n_done >= num_batches:
            break
        batch = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        task_losses = compute_task_losses(outputs, batch, loss_weights, loss_config)
        loss = sum(task_losses)
        loss.backward()
        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
        n_done += 1

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    all_scores = torch.cat([f.flatten() for f in fisher.values()])
    k = max(1, int(ratio * all_scores.numel()))
    score_threshold = torch.topk(all_scores, k, largest=True).values.min()

    masks = {name: (f >= score_threshold).float() for name, f in fisher.items()}
    kept = sum(m.sum().item() for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    logger.info(
        "Child-Tuning: keeping %.1f%% of encoder parameters trainable (%d / %d), "
        "estimated from %d calibration batches",
        100.0 * kept / total, int(kept), total, n_done,
    )
    return masks


def apply_child_tuning_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Zero out gradients for encoder parameters outside the child network.
    Call after loss.backward(), before optimizer.step()."""
    for name, param in model.named_parameters():
        mask = masks.get(name)
        if mask is not None and param.grad is not None:
            param.grad.mul_(mask)


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
    configure_logging(output_dir)

    if args.domain is None and not (args.train_path and args.dev_path and args.test_path):
        raise ValueError(
            "Either --domain or all three of --train_path/--dev_path/--test_path must be given."
        )

    domain_arg = args.domain or "restaurant"
    domain_key = DOMAIN_KEY[domain_arg]
    category_descriptions = mapper.CATEGORY_DESCRIPTIONS.get(domain_key, CATEGORY_DESCRIPTIONS_VI)
    is_raw_text_domain = domain_arg in RAW_TEXT_DOMAIN_ROOTS

    if is_raw_text_domain:
        raw_root = Path(args.raw_root) if args.raw_root else (
            Path(__file__).resolve().parent / RAW_TEXT_DOMAIN_ROOTS[domain_arg]
        )
        train_path = args.train_path or str(raw_root / "Train.txt")
        dev_path = args.dev_path or str(raw_root / "Dev.txt")
        test_path = args.test_path or str(raw_root / "Test.txt")
        train_examples = parse_dataset(train_path)
        dev_examples = parse_dataset(dev_path)
        test_examples = parse_dataset(test_path)
    else:
        pair_root = Path(args.absa_llms_root) / domain_key
        train_path = args.train_path or str(pair_root / "Train.csv")
        dev_path = args.dev_path or str(pair_root / "Dev.csv")
        test_path = args.test_path or str(pair_root / "Test.csv")
        train_examples = load_pair_split(domain_key, train_path)
        dev_examples = load_pair_split(domain_key, dev_path)
        test_examples = load_pair_split(domain_key, test_path)

    categories = validate_categories(train_examples, dev_examples, test_examples)

    print_dataset_stats("train", train_examples)
    print_dataset_stats("dev", dev_examples)
    print_dataset_stats("test", test_examples)
    logger.info("Categories: %s", categories)

    if args.segmenter is None:
        # ABSA_LLMs' Pair CSVs are already cleaned/tokenized per-domain (Mother
        # and Technology are already pyvi-segmented); re-segmenting risks
        # double-processing, so default to no-op for those and keep pyvi only
        # for the raw-text domains (restaurant, hotel).
        seg_name = "pyvi" if is_raw_text_domain else "none"
    else:
        seg_name = args.segmenter
    segmenter = build_segmenter(seg_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = ACSADataset(train_examples, categories)
    dev_ds = ACSADataset(dev_examples, categories)
    test_ds = ACSADataset(test_examples, categories)

    collate = make_collate_fn(tokenizer, segmenter, args.max_length)
    if args.oversample_rare_categories:
        oversample_weights = compute_oversample_weights(train_ds, power=args.oversample_power)
        train_sampler = WeightedRandomSampler(
            weights=oversample_weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
    else:
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
        desc = category_descriptions.get(cat, cat.replace("#", " ").replace("&", " và "))
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
        category_self_attention=args.category_self_attention,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    logger.info("Device: %s", device)
    if device.type == "cpu":
        logger.warning("Training on CPU -- expect this to be very slow for a PhoBERT-scale encoder.")

    loss_weights = compute_loss_weights(train_ds)
    logger.info("ACD pos weights: %s", loss_weights.acd_pos_weight.tolist())
    logger.info("Sentiment class weights [pos, neu, neg]: %s", loss_weights.sent_class_weight.tolist())
    logger.info("Joint class weights [none, pos, neu, neg]: %s", loss_weights.joint_class_weight.tolist())

    loss_config = LossConfig(
        acd_loss_fn=args.acd_loss_fn,
        sent_loss_fn=args.sent_loss_fn,
        joint_loss_fn=args.joint_loss_fn,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
    )
    logger.info(
        "Loss config: acd=%s sent=%s joint=%s focal_gamma=%.2f label_smoothing=%.2f%s",
        loss_config.acd_loss_fn, loss_config.sent_loss_fn, loss_config.joint_loss_fn,
        loss_config.focal_gamma, loss_config.label_smoothing,
        f" asl_gamma_pos={loss_config.asl_gamma_pos:.2f} asl_gamma_neg={loss_config.asl_gamma_neg:.2f} "
        f"asl_clip={loss_config.asl_clip:.2f}" if loss_config.acd_loss_fn == "asl" else "",
    )

    optimizer = build_optimizer(
        model, args.encoder_lr, args.head_lr, args.gate_lr, args.weight_decay, args.llrd_decay
    )

    child_masks = None
    if args.child_tuning:
        child_masks = estimate_child_tuning_mask(
            model, train_loader, loss_weights, loss_config, device,
            num_batches=args.child_tuning_calib_batches, ratio=args.child_tuning_ratio,
        )

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

    # Anomaly tracking across the whole run.
    nonfinite_streak = 0
    total_nonfinite_steps = 0
    total_large_grad_steps = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = np.zeros(4, dtype=np.float64)  # total, acd, sent, joint
        running_steps = 0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")

        for step, batch in enumerate(pbar, start=1):
            global_step += 1
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if gradnorm_optimizer is not None:
                gradnorm_optimizer.zero_grad(set_to_none=True)

            outputs = model(batch["input_ids"], batch["attention_mask"])
            task_losses = compute_task_losses(outputs, batch, loss_weights, loss_config)

            if not all(torch.isfinite(tl) for tl in task_losses):
                total_nonfinite_steps += 1
                nonfinite_streak += 1
                logger.error(
                    "epoch %d step %d (global %d): non-finite task loss acd=%s sent=%s joint=%s -- "
                    "skipping this batch's optimizer step (streak=%d/%d)",
                    epoch, step, global_step,
                    *[tl.item() for tl in task_losses], nonfinite_streak, args.max_nonfinite_steps,
                )
                if nonfinite_streak >= args.max_nonfinite_steps:
                    raise RuntimeError(
                        f"Aborting: {nonfinite_streak} consecutive non-finite losses "
                        f"(epoch {epoch}, step {step}). Check --encoder_lr/--head_lr and input data."
                    )
                continue
            nonfinite_streak = 0

            if args.loss_weighting == "gradnorm":
                assert gradnorm is not None and gradnorm_optimizer is not None
                task_w, grad_w, gradnorm_obj = gradnorm.compute_weight_gradient(
                    task_losses, outputs["shared_z"]
                )
                # Network parameters are optimized with current task weights treated as constants.
                total_loss = sum(w.detach() * loss for w, loss in zip(task_w, task_losses))
                total_loss.backward()
                if child_masks is not None:
                    apply_child_tuning_mask(model, child_masks)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
                if child_masks is not None:
                    apply_child_tuning_mask(model, child_masks)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                display_w = fixed.detach().cpu().tolist()

            if not torch.isfinite(grad_norm):
                logger.error(
                    "epoch %d step %d (global %d): non-finite pre-clip grad_norm=%s",
                    epoch, step, global_step, grad_norm.item(),
                )
            elif grad_norm.item() > args.grad_norm_warn_threshold:
                total_large_grad_steps += 1
                logger.warning(
                    "epoch %d step %d (global %d): large pre-clip grad_norm=%.2f (warn threshold=%.1f)",
                    epoch, step, global_step, grad_norm.item(), args.grad_norm_warn_threshold,
                )

            vals = [total_loss.item()] + [x.item() for x in task_losses]
            running += np.asarray(vals)
            running_steps += 1
            pbar.set_postfix(
                loss=f"{running[0]/running_steps:.4f}",
                acd=f"{running[1]/running_steps:.3f}",
                sent=f"{running[2]/running_steps:.3f}",
                joint=f"{running[3]/running_steps:.3f}",
                w="/".join(f"{x:.2f}" for x in display_w),
                gate=f"{outputs['gate_alpha'].item():.4f}",
                grad=f"{grad_norm.item():.2f}",
            )

        if running_steps == 0:
            raise RuntimeError(f"Epoch {epoch}: every batch had a non-finite loss; aborting.")

        if args.loss_weighting == "gradnorm":
            w = gradnorm.normalized_weights().detach().cpu()
            if w.min().item() < 0.1 or w.max().item() > 2.5:
                logger.warning(
                    "epoch %d: GradNorm task weights [acd, sent, joint] look imbalanced: %s "
                    "(one task may be starved or dominating)",
                    epoch, [round(x, 3) for x in w.tolist()],
                )

        # High-precision visibility into the ACD->sentiment gate: at the
        # default 2-decimal display it looked permanently stuck at 0.50 across
        # entire runs, which is either dead capacity or real (if slow) movement
        # invisible at that precision -- this exposes which one it actually is.
        gate_alpha_val = torch.sigmoid(model.raw_gate_alpha).item()
        logger.info(
            "epoch %d: gate_alpha=%.6f (raw=%.6f)", epoch, gate_alpha_val, model.raw_gate_alpha.item()
        )

        # Tune the ACSA presence threshold on dev for each epoch.
        dev_raw = collect_outputs(model, dev_loader, device)
        if args.per_category_threshold:
            threshold = tune_thresholds_per_category(
                dev_raw, len(categories), min_support=args.threshold_min_support
            )
            dev_metrics, _ = compute_metrics(dev_raw, threshold)
        else:
            threshold, dev_metrics, _ = tune_threshold(dev_raw)
        score = dev_metrics["acsa_f1_micro"]
        threshold_display = float(np.mean(threshold))

        if epoch >= 2 and score == 0.0:
            logger.warning(
                "epoch %d: dev ACSA micro-F1 is exactly 0.0 -- the model may not be learning "
                "(check learning rates, loss weights, or data labels)",
                epoch,
            )

        epoch_record = {
            "epoch": epoch,
            "train_total_loss": float(running[0] / max(1, running_steps)),
            "train_acd_loss": float(running[1] / max(1, running_steps)),
            "train_sent_loss": float(running[2] / max(1, running_steps)),
            "train_joint_loss": float(running[3] / max(1, running_steps)),
            "nonfinite_steps_total_so_far": total_nonfinite_steps,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d: dev ACSA micro-F1=%.4f, macro-F1=%.4f, threshold%s=%.2f, exact=%.4f",
            epoch, score, dev_metrics["acsa_f1_macro"],
            " (mean)" if args.per_category_threshold else "", threshold_display,
            dev_metrics["acsa_exact_match"],
        )

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            patience_counter = 0
            save_checkpoint(
                checkpoint_path, model, args, categories, threshold, dev_metrics
            )
            logger.info("  -> saved new best checkpoint: %s", checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping after epoch %d; best epoch=%d", epoch, best_epoch)
                break

    if total_nonfinite_steps or total_large_grad_steps:
        logger.warning(
            "Training anomaly summary: %d batches skipped for non-finite loss, "
            "%d steps with large pre-clip grad_norm (> %.1f). See %s for full detail.",
            total_nonfinite_steps, total_large_grad_steps, args.grad_norm_warn_threshold,
            output_dir / "train.log",
        )

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Final evaluation with the best dev checkpoint and its tuned threshold.
    # ------------------------------------------------------------------
    # weights_only=False: this checkpoint was written by save_checkpoint() earlier
    # in this same run, not loaded from an untrusted source. Needed explicitly
    # since PyTorch >=2.6 defaults torch.load to weights_only=True, which refuses
    # to unpickle the numpy array stored here when --per_category_threshold is on.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["threshold"]  # scalar (global) or np.ndarray (per-category)
    model.eval()

    dev_raw = collect_outputs(model, dev_loader, device)
    dev_metrics, dev_pred = compute_metrics(dev_raw, best_threshold)
    test_raw = collect_outputs(model, test_loader, device)
    test_metrics, test_pred = compute_metrics(test_raw, best_threshold)

    results = {
        "best_epoch": best_epoch,
        "threshold": best_threshold.tolist() if isinstance(best_threshold, np.ndarray) else best_threshold,
        "categories": categories,
        "dev": dev_metrics,
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_predictions(output_dir / "dev_predictions.jsonl", dev_raw, dev_pred, categories)
    write_predictions(output_dir / "test_predictions.jsonl", test_raw, test_pred, categories)

    logger.info("=== BEST MODEL ===")
    logger.info("Best epoch: %d", best_epoch)
    if isinstance(best_threshold, np.ndarray):
        logger.info("Dev-tuned per-category thresholds (mean=%.2f): %s", float(best_threshold.mean()), dict(zip(categories, best_threshold.round(2).tolist())))
    else:
        logger.info("Dev-tuned threshold: %.2f", best_threshold)
    logger.info("=== TEST ===")
    for key, value in test_metrics.items():
        logger.info("%-32s: %.6f", key, value)
    logger.info("Artifacts written to: %s", output_dir.resolve())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Category-conditioned multi-task ACSA trainer")
    p.add_argument(
        "--domain",
        choices=sorted(DOMAIN_KEY),
        default=None,
        help="Selects data paths + category descriptions automatically. "
        "'restaurant' and 'hotel' train from their own original *_ABSA/*.txt files "
        "(raw text; restaurant is the validated 77.43 F1 baseline). 'restaurant_clean' / "
        "'hotel_clean' train on data/Pair/<Domain>'s clean_doc()-preprocessed text instead, "
        "for an A/B against the raw-text version of the same domain -- note hotel_clean's "
        "train/dev/test split is NOT sample-identical to hotel's (ABSA_LLMs re-shuffled "
        "Hotel's data; verified same underlying pool via aggregate label distribution, but "
        "restaurant_clean IS row-for-row identical to restaurant, verified directly). "
        "The remaining domains load ABSA_LLMs' cleaned data/Pair/<Domain>/{Train,Dev,Test}.csv. "
        "Omit and pass --train_path/--dev_path/--test_path explicitly for full control.",
    )
    p.add_argument(
        "--absa_llms_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "Pair"),
        help="Root containing <Domain>/{Train,Dev,Test}.csv, used for every --domain except "
        "restaurant/hotel (a local copy of ABSA_LLMs/data/Pair, self-contained so the whole "
        "ACSA folder can be uploaded to Colab/Kaggle as-is).",
    )
    p.add_argument(
        "--raw_root",
        type=str,
        default=None,
        help="Root containing {Train,Dev,Test}.txt for a raw-text --domain (restaurant/hotel). "
        "Defaults to Res_ABSA/ or Hotel_ABSA/ respectively, next to this script.",
    )
    p.add_argument("--train_path", type=str, default=None, help="Overrides the path --domain would resolve to.")
    p.add_argument("--dev_path", type=str, default=None, help="Overrides the path --domain would resolve to.")
    p.add_argument("--test_path", type=str, default=None, help="Overrides the path --domain would resolve to.")
    p.add_argument("--output_dir", type=str, default="outputs/mtl_acsa")

    p.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    p.add_argument(
        "--segmenter",
        choices=["none", "pyvi"],
        default=None,
        help="Default: pyvi for restaurant (raw text), none for the other domains (already pre-tokenized).",
    )
    p.add_argument("--max_length", type=int, default=160)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--adapter_dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--category_self_attention", action="store_true",
        help="Add a self-attention layer among the K category representations after the text "
        "cross-attention, so categories can share signal with each other (e.g. rare categories "
        "borrowing context from correlated, well-observed ones). Default off -- new architecture "
        "change, not yet validated.",
    )

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument(
        "--gate_lr", type=float, default=None,
        help="LR for the single-scalar ACD->sentiment gate (raw_gate_alpha), given its own "
        "zero-weight-decay optimizer group instead of sharing head_lr/weight_decay with everything "
        "else. Default None: gate_alpha shares head_lr/weight_decay like the rest of the heads, "
        "matching the original script exactly -- it was observed stuck at gate_alpha=0.50 for entire "
        "runs under that scheme, so pass e.g. --gate_lr 5e-3 to opt into the fix.",
    )
    p.add_argument(
        "--llrd_decay", type=float, default=1.0,
        help="Layer-wise LR decay factor for the encoder (1.0 = off, every layer uses encoder_lr). "
        "Values like 0.9-0.95 give the top transformer layer the full encoder_lr and progressively "
        "less to lower layers/embeddings -- a standard trick for fine-tuning pretrained encoders.",
    )
    p.add_argument(
        "--child_tuning", action="store_true",
        help="Child-Tuning_D (Xu et al., EMNLP 2021): before training, estimate each encoder "
        "parameter's importance via Fisher information on a calibration pass, then only update the "
        "top --child_tuning_ratio fraction of encoder parameters every step (the rest get their "
        "gradient zeroed). Protects pretrained encoder knowledge from overfitting on small datasets; "
        "only applied to the encoder, not the freshly-initialized heads/adapters.",
    )
    p.add_argument(
        "--child_tuning_ratio", type=float, default=0.3,
        help="Fraction of encoder parameters kept trainable under --child_tuning (paper default range: 0.2-0.3).",
    )
    p.add_argument(
        "--child_tuning_calib_batches", type=int, default=50,
        help="Number of training batches used to estimate the Fisher-information importance mask "
        "under --child_tuning, run once before the epoch loop starts.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument(
        "--max_nonfinite_steps", type=int, default=20,
        help="Abort training after this many consecutive batches with a non-finite (NaN/Inf) loss.",
    )
    p.add_argument(
        "--grad_norm_warn_threshold", type=float, default=20.0,
        help="Log a warning when a step's pre-clip gradient norm exceeds this (possible exploding gradients).",
    )

    p.add_argument("--loss_weighting", choices=["fixed", "gradnorm"], default="gradnorm")
    p.add_argument("--lambda_acd", type=float, default=1.0)
    p.add_argument("--lambda_sent", type=float, default=1.0)
    p.add_argument("--lambda_joint", type=float, default=0.5)
    p.add_argument("--gradnorm_alpha", type=float, default=1.5)
    p.add_argument("--gradnorm_lr", type=float, default=2.5e-3)

    p.add_argument(
        "--acd_loss_fn", choices=["bce", "focal", "asl"], default="bce",
        help="Loss for the ACD (category presence) head. 'focal' down-weights already-easy "
        "decisions on top of the existing inverse-frequency pos_weight. 'asl' (Asymmetric Loss, "
        "Ben-Baruch et al. 2021) is a multi-label-specific alternative: treats positives and "
        "negatives asymmetrically and hard-discards trivially-easy negatives via probability "
        "shifting -- ignores --acd_pos_weight when selected, since ASL's own asymmetric "
        "mechanism already targets the same imbalance. Not offered for --sent_loss_fn/"
        "--joint_loss_fn since ASL is defined for multi-label (sigmoid) heads, not the "
        "multi-class (softmax) sentiment/joint heads.",
    )
    p.add_argument(
        "--sent_loss_fn", choices=["ce", "focal"], default="ce",
        help="Loss for the sentiment (3-class) head. 'focal' ignores the data-driven inverse-frequency "
        "class weight (like 'asl' ignores acd_pos_weight above) to avoid double-correcting the same "
        "imbalance twice.",
    )
    p.add_argument(
        "--joint_loss_fn", choices=["ce", "focal"], default="ce",
        help="Loss for the joint ACSA (4-class) head. 'focal' ignores its class weight for the same "
        "reason as --sent_loss_fn focal.",
    )
    p.add_argument(
        "--focal_gamma", type=float, default=2.0,
        help="Focusing parameter for any head set to 'focal' (standard default from Lin et al. 2017).",
    )
    p.add_argument(
        "--label_smoothing", type=float, default=0.0,
        help="Label smoothing for sentiment/joint heads set to 'ce' (ignored for 'focal' heads).",
    )
    p.add_argument(
        "--asl_gamma_pos", type=float, default=0.0,
        help="ASL focusing exponent for positive (present) labels, only used with --acd_loss_fn asl. "
        "Paper default 0 (no down-weighting -- positives are the scarce, valuable signal).",
    )
    p.add_argument(
        "--asl_gamma_neg", type=float, default=4.0,
        help="ASL focusing exponent for negative (absent) labels, only used with --acd_loss_fn asl. "
        "Paper default 4 (aggressive down-weighting -- negatives vastly outnumber positives here).",
    )
    p.add_argument(
        "--asl_clip", type=float, default=0.05,
        help="ASL probability-shifting margin, only used with --acd_loss_fn asl. Paper default 0.05.",
    )
    p.add_argument(
        "--per_category_threshold", dest="per_category_threshold", action="store_true", default=False,
        help="Tune a separate ACD presence threshold per category on dev instead of one threshold "
        "shared across all categories. Default off, matching the original script exactly. Pure "
        "post-hoc reranking of already-collected logits, no effect on the trained weights -- "
        "categories with too few dev examples (--threshold_min_support) safely fall back to the "
        "global threshold, so this is a strict improvement or exactly neutral over the single-"
        "threshold behavior, never worse -- pass this flag to opt in.",
    )
    p.add_argument(
        "--no_per_category_threshold", dest="per_category_threshold", action="store_false",
        help="Disable per-category threshold tuning; use a single global threshold instead (the "
        "original behavior before --per_category_threshold existed).",
    )
    p.add_argument(
        "--threshold_min_support", type=int, default=20,
        help="Categories with fewer than this many gold-positive dev examples keep the global "
        "threshold instead of getting their own (only used with --per_category_threshold) -- "
        "tuning on a handful of examples fits noise, not a real operating point.",
    )
    p.add_argument(
        "--oversample_rare_categories", action="store_true",
        help="Sample training examples with a WeightedRandomSampler instead of uniformly: examples "
        "touching a rare category are drawn more often per epoch. A data-side lever (how often the "
        "model sees an example) rather than a loss-side one (how much its gradient counts) -- worth "
        "trying since none of the loss-weighting techniques moved the aggregate ACSA F1 much.",
    )
    p.add_argument(
        "--oversample_power", type=float, default=0.5,
        help="Softening exponent for --oversample_rare_categories (0.5 = sqrt, matching the existing "
        "class-weight softening elsewhere). Higher = more aggressive oversampling of rare categories.",
    )

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
