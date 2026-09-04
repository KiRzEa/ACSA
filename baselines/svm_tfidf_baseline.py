#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistics-based SVM baseline (the "SVM" row in Tables 3/4), ported from
the problem-transformation method of Dang Van Thin et al., "A Transformation
Method for Aspect-Based Sentiment Analysis" (docs/references/baseline_method_paper.pdf,
JCSC 2018): TF-IDF n-gram features + one linear-SVM aspect-detection classifier
per category, followed by one linear-SVM 3-way polarity classifier per category
(trained only on the examples where that category is present).

Runs entirely on CPU in well under a minute even on the largest domain (Beauty),
so unlike the other baselines this does not need a Kaggle GPU session.

Example:
    python3 baselines/svm_tfidf_baseline.py \\
        --train_path Beauty_ABSA/Train.txt --dev_path Beauty_ABSA/Dev.txt \\
        --test_path Beauty_ABSA/Test.txt --output_dir outputs/svm_beauty
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Example, infer_categories, load_examples, micro_prf, write_metrics, write_predictions  # noqa: E402

logger = logging.getLogger("svm_tfidf_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def segment(text: str) -> str:
    return ViTokenizer.tokenize(text)


def build_vectorizer(texts: List[str]) -> TfidfVectorizer:
    # Unigram/bigram/trigram word features, as in the paper (Section 3.3/3.4).
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    vec.fit(texts)
    return vec


def train_and_predict(
    train: List[Example], dev: List[Example], test: List[Example], categories: List[str], seed: int
) -> Tuple[List[List[Tuple[str, str]]], Dict[str, float]]:
    train_dev = train + dev  # SVM is deterministic and needs every labeled example it can get
    texts = [segment(ex.text) for ex in train_dev]
    vectorizer = build_vectorizer(texts)
    X = vectorizer.transform(texts)

    aspect_clfs: Dict[str, LinearSVC] = {}
    polarity_clfs: Dict[str, LinearSVC] = {}
    for cat in categories:
        y_present = [1 if any(c == cat for c, _ in ex.labels) else 0 for ex in train_dev]
        if len(set(y_present)) < 2:
            aspect_clfs[cat] = None  # category never/always present in train+dev -- degenerate, skip
        else:
            clf = LinearSVC(random_state=seed, class_weight="balanced")
            clf.fit(X, y_present)
            aspect_clfs[cat] = clf

        pol_idx = [i for i, ex in enumerate(train_dev) if any(c == cat for c, _ in ex.labels)]
        y_pol = [next(s for c, s in train_dev[i].labels if c == cat) for i in pol_idx]
        if len(set(y_pol)) < 2 or len(pol_idx) < 2:
            polarity_clfs[cat] = None
        else:
            clf = LinearSVC(random_state=seed, class_weight="balanced")
            clf.fit(X[pol_idx], y_pol)
            polarity_clfs[cat] = clf

    test_texts = [segment(ex.text) for ex in test]
    X_test = vectorizer.transform(test_texts)

    predictions: List[List[Tuple[str, str]]] = [[] for _ in test]
    for cat in categories:
        aspect_clf = aspect_clfs[cat]
        if aspect_clf is None:
            continue
        present = aspect_clf.predict(X_test)
        polarity_clf = polarity_clfs[cat]
        if polarity_clf is not None:
            polarities = polarity_clf.predict(X_test)
        else:
            polarities = ["positive"] * len(test)  # degenerate fallback: majority class not learnable
        for i, is_present in enumerate(present):
            if is_present:
                predictions[i].append((cat, polarities[i]))

    metrics = micro_prf([ex.labels for ex in test], predictions)
    return predictions, metrics


def run(args: argparse.Namespace) -> None:
    train = load_examples(args.train_path)
    dev = load_examples(args.dev_path)
    test = load_examples(args.test_path)
    categories = infer_categories(train, dev, test)
    logger.info("Loaded %d train / %d dev / %d test examples, %d categories", len(train), len(dev), len(test), len(categories))

    output_dir = Path(args.output_dir)
    predictions, metrics = train_and_predict(train, dev, test, categories, args.seed)
    logger.info("Test micro P/R/F1: %.2f / %.2f / %.2f", metrics["precision"], metrics["recall"], metrics["f1"])

    write_predictions(output_dir / "test_predictions.jsonl", test, predictions)
    write_metrics(output_dir / "test_metrics.json", {**metrics, "seed": args.seed, "categories": categories})
    logger.info("Wrote predictions/metrics to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TF-IDF + linear-SVM problem-transformation ACSA baseline")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--dev_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
