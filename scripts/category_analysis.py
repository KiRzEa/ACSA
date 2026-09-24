"""Per-category analysis of CAGE vs an independent-head baseline (same PhoBERT encoder).

Question 1 (semantic overlap): do categories whose training usage overlaps more with another category gain more?
Question 2 (rarity): does CAGE help rare categories more than frequent ones?

Per-category joint F1 (pair = category + polarity): tp = category present in gold and pred with the same polarity;
fp = predicted pair not matched; fn = gold pair not matched.  Gain = mean CAGE F1 over seeds - baseline F1.
Overlap of category c = max over c' != c of the cosine between the TF-IDF centroids of the training reviews that
carry c and c' (model-free, measures how much two categories are used on the same kind of content).

Usage: python3 scripts/category_analysis.py [--baseline bert_{d}_phobert] [--min_test 10] [--json out.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_mtl_acsa_v2 import parse_dataset  # noqa: E402

DOMAINS = {"restaurant": "Res_ABSA", "hotel": "Hotel_ABSA", "phone": "Phone_ABSA",
           "education": "Education_ABSA", "beauty": "Beauty_ABSA"}


def per_category_f1(path):
    tp, fp, fn = {}, {}, {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g = {x["category"]: x["sentiment"] for x in r["gold"]}
        p = {x["category"]: x["sentiment"] for x in r["prediction"]}
        for c in set(g) | set(p):
            ok = c in g and c in p and g[c] == p[c]
            tp[c] = tp.get(c, 0) + ok
            fp[c] = fp.get(c, 0) + (c in p and not ok)
            fn[c] = fn.get(c, 0) + (c in g and not ok)
    cats = set(tp) | set(fp) | set(fn)
    return {c: (200 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) if 2 * tp[c] + fp[c] + fn[c] else 0.0,
                tp[c] + fn[c]) for c in cats}  # (F1, gold count)


def overlap_and_support(domain):
    train = parse_dataset(ROOT / DOMAINS[domain] / "Train.txt")
    docs = {}
    for ex in train:
        for c, _ in ex.labels:
            docs.setdefault(c, []).append(ex.text)
    cats = sorted(docs)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([ex.text for ex in train])
    cent = np.vstack([np.asarray(vec.transform(docs[c]).mean(axis=0)).ravel() for c in cats])
    cent = np.nan_to_num(cent.astype(np.float64))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    sim = np.einsum("id,jd->ij", cent, cent)
    np.fill_diagonal(sim, -1)
    return {c: (float(sim[i].max()), cats[int(sim[i].argmax())], len(docs[c])) for i, c in enumerate(cats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cage", default="outputs/cage_{d}_fixed")
    ap.add_argument("--baseline", default="outputs/bert_{d}_phobert")
    ap.add_argument("--min_test", type=int, default=10)
    ap.add_argument("--json", default="paper/error_analysis/category_analysis.json")
    a = ap.parse_args()
    rows = []
    for d in DOMAINS:
        base = per_category_f1(ROOT / a.baseline.format(d=d) / "test_predictions.jsonl")
        seeds = sorted((ROOT / a.cage.format(d=d)).glob("seed_*/test_predictions.jsonl"))
        runs = [per_category_f1(s) for s in seeds]
        ov = overlap_and_support(d)
        for c, (ovl, nb, ntr) in ov.items():
            if c not in base or base[c][1] < a.min_test:
                continue
            cage_f1 = np.mean([r.get(c, (0.0, 0))[0] for r in runs])
            rows.append({"domain": d, "category": c, "train": ntr, "test": base[c][1], "overlap": ovl, "nearest": nb,
                         "base_f1": base[c][0], "cage_f1": float(cage_f1), "gain": float(cage_f1 - base[c][0]),
                         "n_seeds": len(runs)})
    out = {"rows": rows}

    def corr(x, y):
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)

    print(f"{len(rows)} categories with >= {a.min_test} test gold pairs")
    for d in [*DOMAINS, "ALL"]:
        rs = [r for r in rows if d == "ALL" or r["domain"] == d]
        if len(rs) < 5:
            continue
        g = [r["gain"] for r in rs]
        ro, po = corr([r["overlap"] for r in rs], g)
        rt, pt = corr([np.log(r["train"]) for r in rs], g)
        rso, _ = corr([r["overlap"] for r in rs], [np.log(r["train"]) for r in rs])
        print(f"{d:10s} n={len(rs):3d} (overlap~support {rso:+.2f}) mean gain {np.mean(g):+.2f} | spearman(gain, overlap) {ro:+.2f} (p={po:.3f})"
              f" | spearman(gain, log train) {rt:+.2f} (p={pt:.3f})")
        out[f"corr/{d}"] = {"n": len(rs), "gain_overlap": [ro, po], "gain_logtrain": [rt, pt], "mean_gain": float(np.mean(g))}
    for name, key, cuts in [("rarity", "train", (20, 100)), ("overlap", "overlap", None)]:
        print(f"\n{name} bins (pooled, gain = CAGE - baseline F1)")
        vals = np.array([r[key] for r in rows])
        edges = cuts if cuts else tuple(np.quantile(vals, [1 / 3, 2 / 3]))
        for lo, hi, lab in [(-np.inf, edges[0], "low"), (edges[0], edges[1], "mid"), (edges[1], np.inf, "high")]:
            g = [r["gain"] for r in rows if lo <= r[key] < hi]
            if g:
                print(f"  {lab:4s} [{lo if lo>-1e9 else '-inf'}, {hi if hi<1e9 else 'inf'}) n={len(g):3d} mean gain {np.mean(g):+.2f}  median {np.median(g):+.2f}")
                out[f"bin/{name}/{lab}"] = {"n": len(g), "mean_gain": float(np.mean(g)), "edges": [float(edges[0]), float(edges[1])]}
    Path(ROOT / a.json).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print("wrote", a.json)


if __name__ == "__main__":
    main()
