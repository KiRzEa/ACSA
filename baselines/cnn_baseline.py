#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN baseline (the "CNN" row in Tables 3/4): the convolutional half of the
BiLSTM-CNN model in notebooks/legacy/bilstm_cnn_prototype.ipynb, with the
BiLSTM branch removed -- trainable word embeddings -> parallel Conv1d(k=2,3,4)
-> global max/average pooling -> concat -> one 4-way softmax head per category
(NONE / positive / neutral / negative), trained jointly with summed per-category
cross-entropy. Reimplemented in PyTorch (the prototype used Keras/TF) to match
the rest of this repo's stack and CLI conventions (train_mtl_acsa_v2.py).

Multi-seed: pass --seeds "42,123,2024" to train 3 independent runs and report
best-of-3 test F1, matching the reporting convention used everywhere else in
the paper (Section 4.3 / Tables 3-4).

Example:
    python3 baselines/cnn_baseline.py \\
        --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/cnn_beauty \\
        --seeds 42,123,2024
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvi import ViTokenizer
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Example, infer_categories, load_examples, micro_prf, write_metrics, write_predictions  # noqa: E402

logger = logging.getLogger("cnn_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

PAD, UNK = "<pad>", "<unk>"
LABELS = ["NONE", "positive", "neutral", "negative"]  # class 0 = category absent
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segment(text: str) -> List[str]:
    return ViTokenizer.tokenize(text).split()


def build_vocab(examples: Sequence[Example], min_freq: int) -> Dict[str, int]:
    counts = Counter()
    for ex in examples:
        counts.update(segment(ex.text))
    vocab = {PAD: 0, UNK: 1}
    for tok, c in counts.most_common():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab[UNK]) for t in tokens[:max_len]]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class ACSADataset(Dataset):
    def __init__(self, examples: Sequence[Example], vocab: Dict[str, int], categories: List[str], max_len: int):
        self.examples = examples
        self.vocab = vocab
        self.categories = categories
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        ids = encode(segment(ex.text), self.vocab, self.max_len)
        label_by_cat = {c: s for c, s in ex.labels}
        targets = [LABEL2IDX.get(label_by_cat.get(cat, "NONE"), 0) for cat in self.categories]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class CNNBaseline(nn.Module):
    def __init__(self, vocab_size: int, num_categories: int, emb_dim: int = 300, num_filters: int = 128, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.spatial_dropout = nn.Dropout2d(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, kernel_size=k, padding=k // 2) for k in (2, 3, 4)]
        )
        pooled_dim = num_filters * 2 * len(self.convs)  # max + avg pool per conv branch
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(pooled_dim, len(LABELS)) for _ in range(num_categories)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        emb = self.embedding(x)  # (B, T, C)
        emb = emb.unsqueeze(1)  # (B, 1, T, C) so Dropout2d drops whole embedding channels
        emb = self.spatial_dropout(emb).squeeze(1)  # (B, T, C)
        emb = emb.transpose(1, 2)  # (B, C, T) for Conv1d
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))  # (B, F, T')
            pooled.append(h.max(dim=2).values)
            pooled.append(h.mean(dim=2))
        feat = self.dropout(torch.cat(pooled, dim=1))
        return [head(feat) for head in self.heads]


def train_one_seed(
    train: List[Example], dev: List[Example], test: List[Example], categories: List[str], args: argparse.Namespace, seed: int
) -> Tuple[Dict[str, float], List[List[Tuple[str, str]]]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    vocab = build_vocab(train, args.min_freq)
    logger.info("[seed %d] vocab size: %d", seed, len(vocab))

    train_ds = ACSADataset(train, vocab, categories, args.max_len)
    dev_ds = ACSADataset(dev, vocab, categories, args.max_len)
    test_ds = ACSADataset(test, vocab, categories, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size)

    model = CNNBaseline(len(vocab), len(categories), dropout=args.dropout).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate)

    def run_eval(loader: DataLoader, examples: List[Example]) -> Tuple[Dict[str, float], List[List[Tuple[str, str]]]]:
        model.eval()
        predictions: List[List[Tuple[str, str]]] = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                pred_idx = torch.stack([l.argmax(dim=1) for l in logits], dim=1).cpu().numpy()
                for row in pred_idx:
                    predictions.append(
                        [(categories[c], LABELS[row[c]]) for c in range(len(categories)) if row[c] != 0]
                    )
        metrics = micro_prf([ex.labels for ex in examples], predictions)
        return metrics, predictions

    best_dev_f1, best_state, patience_left = -1.0, None, args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = sum(F.cross_entropy(logits[c], y[:, c]) for c in range(len(categories)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_metrics, _ = run_eval(dev_loader, dev)
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
    test_metrics, test_predictions = run_eval(test_loader, test)
    return test_metrics, test_predictions


def run(args: argparse.Namespace) -> None:
    train = load_examples(args.train_path)
    dev = load_examples(args.dev_path)
    test = load_examples(args.test_path)
    categories = infer_categories(train, dev, test)
    logger.info("Loaded %d train / %d dev / %d test examples, %d categories", len(train), len(dev), len(test), len(categories))

    output_dir = Path(args.output_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed_results = []
    for seed in seeds:
        metrics, predictions = train_one_seed(train, dev, test, categories, args, seed)
        logger.info("[seed %d] test P/R/F1: %.2f / %.2f / %.2f", seed, metrics["precision"], metrics["recall"], metrics["f1"])
        per_seed_results.append({"seed": seed, "metrics": metrics, "predictions": predictions})

    best = max(per_seed_results, key=lambda r: r["metrics"]["f1"])
    logger.info("Best of %d seeds: seed=%d test_f1=%.2f", len(seeds), best["seed"], best["metrics"]["f1"])

    write_predictions(output_dir / "test_predictions.jsonl", test, best["predictions"])
    summary = {
        "best_seed": best["seed"],
        "best": best["metrics"],
        "per_seed": [{"seed": r["seed"], **r["metrics"]} for r in per_seed_results],
        "categories": categories,
    }
    write_metrics(output_dir / "multi_seed_summary.json", summary)
    logger.info("Wrote predictions/summary to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pure-CNN (BiLSTM branch removed) ACSA baseline")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--dev_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seeds", type=str, default="42,123,2024")
    p.add_argument("--max_len", type=int, default=120)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
