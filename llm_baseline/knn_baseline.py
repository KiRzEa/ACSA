#!/usr/bin/env python3
"""Retrieval-only control for the retrieval few-shot LLM baseline: no LLM, only the labels of the K nearest training reviews
(same char n-gram TF-IDF neighbours). A category is predicted when the similarity-weighted share of neighbours carrying it
reaches a threshold tuned on Dev; its polarity is the similarity-weighted majority among those neighbours.
Shows how much of the retrieval-few-shot F1 comes from the neighbours' labels alone.
Usage: python3 llm_baseline/knn_baseline.py [--domains education hotel ...] [--k 10]"""
import argparse, contextlib, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "llm_baseline"))
from run_llm_acsa import DATA, load  # noqa: E402
from evaluate import evaluate_jsonl_per_category  # noqa: E402


def neighbours(train, queries, k):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=True)
    xt = vec.fit_transform([e.text for e in train]); xq = vec.transform([e.text for e in queries])
    sims = (xq @ xt.T).tocsr(); out = []
    for i in range(len(queries)):
        row = sims.getrow(i); order = row.data.argsort()[::-1][:k]
        out.append([(int(row.indices[j]), float(row.data[j])) for j in order])
    return out


def predict(train, nn, thr):
    preds = []
    for nb in nn:
        tot = sum(s for _, s in nb) or 1.0; score = defaultdict(float); pol = defaultdict(lambda: defaultdict(float))
        for j, s in nb:
            for c, sv in train[j].labels:
                score[c] += s / tot; pol[c][sv] += s
        preds.append({c: max(pol[c], key=pol[c].get) for c, v in score.items() if v >= thr})
    return preds


def f1(examples, preds, path):
    with path.open("w", encoding="utf-8") as f:
        for e, p in zip(examples, preds):
            f.write(json.dumps({"id": e.sample_id, "text": e.text, "gold": [{"category": c, "sentiment": s} for c, s in e.labels],
                                "prediction": [{"category": c, "sentiment": s} for c, s in p.items()]}, ensure_ascii=False) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_jsonl_per_category(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--domains", nargs="+", default=list(DATA)); ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(); res = {}
    for d in a.domains:
        tr, dv, te = load(d, "Train"), load(d, "Dev"), load(d, "Test")
        nd, nt = neighbours(tr, dv, a.k), neighbours(tr, te, a.k)
        best = max(((f1(dv, predict(tr, nd, t), Path("/tmp/knn_dev.jsonl"))[2], t) for t in np.arange(0.2, 0.95, 0.05)))
        out = ROOT / "outputs_llm" / f"knn_{d}_k{a.k}"; out.mkdir(parents=True, exist_ok=True)
        P, R, F = f1(te, predict(tr, nt, best[1]), out / "test_predictions.jsonl")
        res[d] = {"dev_F1": round(best[0], 2), "threshold": round(float(best[1]), 2), "P": round(P, 2), "R": round(R, 2), "F1": round(F, 2)}
        print(d, res[d], flush=True)
    (ROOT / "outputs_llm" / f"knn_k{a.k}_summary.json").write_text(json.dumps(res, indent=2))
