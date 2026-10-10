"""Controlled re-analysis of per-category CAGE gains (reviewer Major Comment 4, Section 5.4.5 / sec:category-gains).

Builds on scripts/category_analysis.py (same rows: categories with >= 5 gold test pairs, gain = mean CAGE F1 over
5 seeds - mean PhoBERT F1 over 5 seeds, support = # training reviews carrying the category, overlap = max TF-IDF
centroid cosine to another category).  Adds:

  1. Reproduction of the reported Spearman correlations (gain~log support, gain~overlap, overlap~log support, n=74),
     recomputed from the raw test_predictions.jsonl files and checked row-by-row against category_analysis.json.
  2. OLS with domain fixed effects and HC3 standard errors (implemented in numpy; statsmodels is not installed; HC3 is
     self-checked against the exact leave-one-out identity), t(n-k) reference distribution.
  3. Independent description-similarity measures: for each category, the max cosine between its natural-language
     description (the category_texts.json the trained CAGE models actually used) and any OTHER category's description
     in the same domain, embedded by encoders that were NOT trained in this project:
        phobert_mean  frozen vinai/phobert-base-v2, PyVi-segmented text (as in training), mean of last hidden states
                      over non-special tokens  [PRIMARY]
        phobert_cls   same encoder, first-token (<s>) state, i.e. the pooling CAGE's query uses, but frozen [sensitivity]
        xlmr_mean     frozen xlm-roberta-base, raw text, mean pooling [secondary; multilingual MLM, not a sentence model]
        nllb_mean     frozen facebook/nllb-200-distilled-600M encoder (src vie_Latn), raw text, mean pooling
                      [secondary; translation-trained encoder]
        jaccard       Jaccard over PyVi word tokens (lower-cased, punctuation dropped) [lexical sanity reference]
     No multilingual sentence-embedding model (e.g. paraphrase-multilingual-*, LaBSE, multilingual-e5) is in the local
     HF cache, so none is used.
  4. Rank-based checks: domain-demeaned Spearman (with within-domain permutation p-values), partial Spearman
     controlling for log support (and for log support + domain), an exploratory log support x similarity interaction,
     and an exploratory rare-category (train <= 100) subset.

Runs fully offline.  Usage:
    HF_HUB_OFFLINE=1 python3 scripts/category_gain_controls.py [--json paper/error_analysis/category_gain_controls.json]
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse, json, re, sys, warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# numpy + macOS Accelerate emits spurious "divide by zero/overflow/invalid value encountered in matmul" warnings once
# torch is loaded; every matmul result below is asserted finite and HC3 is cross-checked against a LAPACK lstsq path.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from category_analysis import DOMAINS, overlap_and_support, per_category_f1, per_category_f1_multiseed  # noqa: E402

DOMAIN_ORDER = list(DOMAINS)  # restaurant (reference level), hotel, phone, education, beauty
MIN_TEST = 5                  # paper: ">= 5 gold test pairs" (the stored category_analysis.json was built with this)
RARE_MAX = 100                # paper's "20 to 100 training examples" band (no tested category has < 20)
N_PERM = 10000
SEED = 0
REPORTED = {"gain_logtrain": -0.41, "gain_overlap": -0.05, "overlap_logtrain": 0.69, "n": 74, "p_overlap": 0.64}
MEASURES = ["phobert_mean", "phobert_cls", "xlmr_mean", "nllb_mean", "jaccard"]
CENTERED = ["phobert_mean", "xlmr_mean", "nllb_mean"]  # robustness: mean-centred embeddings (anisotropy)
PRIMARY = "phobert_mean"


# ----------------------------------------------------------------------------------------------------------------
# 1. rows (identical construction to scripts/category_analysis.py with --min_test 5)
# ----------------------------------------------------------------------------------------------------------------
def build_rows(out_dir="outputs", cage="outputs/cage_{d}_fixed", baseline="bert_{d}_phobert", min_test=MIN_TEST):
    rows = []
    for d in DOMAIN_ORDER:
        base, n_base = per_category_f1_multiseed(ROOT / out_dir, baseline.format(d=d))
        assert n_base == 5, (d, n_base)
        seeds = sorted((ROOT / cage.format(d=d)).glob("seed_*/test_predictions.jsonl"))
        runs = [per_category_f1(s) for s in seeds]
        assert len(runs) == 5, (d, len(runs))
        for c, (ovl, nb, ntr) in overlap_and_support(d).items():
            if c not in base or base[c][1] < min_test:
                continue
            cage_f1 = float(np.mean([r.get(c, (0.0, 0))[0] for r in runs]))
            rows.append({"domain": d, "category": c, "train": ntr, "test": base[c][1], "overlap": ovl,
                         "overlap_nearest": nb, "base_f1": base[c][0], "cage_f1": cage_f1,
                         "gain": cage_f1 - base[c][0]})
    return rows


def check_against_stored(rows, path):
    stored = {(r["domain"], r["category"]): r for r in json.loads(Path(path).read_text())["rows"]}
    mine = {(r["domain"], r["category"]): r for r in rows}
    out = {"stored_file": str(Path(path).relative_to(ROOT)), "same_category_set": set(stored) == set(mine),
           "n_stored": len(stored), "n_recomputed": len(mine)}
    common = set(stored) & set(mine)
    for k in ["gain", "overlap", "base_f1", "cage_f1"]:
        out[f"max_abs_diff_{k}"] = max(abs(stored[c][k] - mine[c][k]) for c in common)
    for k in ["train", "test"]:
        out[f"n_mismatch_{k}"] = sum(stored[c][k] != mine[c][k] for c in common)
    out["n_mismatch_nearest"] = sum(stored[c]["nearest"] != mine[c]["overlap_nearest"] for c in common)
    return out


# ----------------------------------------------------------------------------------------------------------------
# 2. description similarity
# ----------------------------------------------------------------------------------------------------------------
def load_category_texts(cage="outputs/cage_{d}_fixed"):
    texts, identical = {}, {}
    for d in DOMAIN_ORDER:
        files = sorted((ROOT / cage.format(d=d)).glob("seed_*/category_texts.json"))
        ref = (ROOT / cage.format(d=d) / "seed_42" / "category_texts.json").read_text(encoding="utf-8")
        identical[d] = {"n_seed_files": len(files), "all_identical": all(f.read_text(encoding="utf-8") == ref for f in files)}
        texts[d] = json.loads(ref)
    return texts, identical


def embed(texts_by_domain, kind):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from pyvi import ViTokenizer

    torch.manual_seed(SEED)
    if kind.startswith("phobert"):
        name = "vinai/phobert-base-v2"
        tok = AutoTokenizer.from_pretrained(name, use_fast=False)  # same tokenizer call as train_mtl_acsa_v2.py
        model = AutoModel.from_pretrained(name)
        prep = ViTokenizer.tokenize  # build_segmenter("pyvi") in train_mtl_acsa_v2.py
    elif kind == "xlmr_mean":
        name = "xlm-roberta-base"
        tok, model, prep = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name), (lambda s: s)
    elif kind == "nllb_mean":
        from transformers import AutoModelForSeq2SeqLM
        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name, src_lang="vie_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(name).get_encoder()
        prep = lambda s: s  # noqa: E731
    else:
        raise ValueError(kind)
    model.eval()
    info = {"model": name, "commit": getattr(model.config, "_commit_hash", None),
            "input": "PyVi-segmented" if kind.startswith("phobert") else "raw",
            "pooling": "first token (<s>)" if kind == "phobert_cls" else "mean over non-special tokens"}
    out = {}
    with torch.no_grad():
        for d, td in texts_by_domain.items():
            cats = list(td)
            enc = tok([prep(td[c]) for c in cats], padding=True, truncation=True, max_length=48,
                      return_tensors="pt", return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            enc.pop("token_type_ids", None)
            h = model(**enc).last_hidden_state.double()
            if kind == "phobert_cls":
                e = h[:, 0, :]
            else:
                m = (enc["attention_mask"].bool() & ~special.bool()).double().unsqueeze(-1)
                e = (h * m).sum(1) / m.sum(1)
            out[d] = (cats, e.numpy())
    return out, info


def word_tokens(s):
    from pyvi import ViTokenizer
    return {t.lower() for t in ViTokenizer.tokenize(s).split() if re.search(r"\w", t)}


def max_other_similarity(texts_by_domain, emb=None):
    """{(domain, cat): (max sim to another category in the same domain, argmax category)}; all categories of the
    domain (not only the analysed ones) are candidate neighbours."""
    res = {}
    for d, td in texts_by_domain.items():
        cats = list(td)
        if emb is None:  # Jaccard
            toks = [word_tokens(td[c]) for c in cats]
            S = np.array([[len(a & b) / len(a | b) for b in toks] for a in toks], dtype=float)
        else:
            cats_e, E = emb[d]
            assert cats_e == cats
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            S = E @ E.T
        assert np.isfinite(S).all() and np.abs(S).max() <= 1 + 1e-9
        np.fill_diagonal(S, -np.inf)
        for i, c in enumerate(cats):
            res[(d, c)] = (float(S[i].max()), cats[int(S[i].argmax())])
    return res


# ----------------------------------------------------------------------------------------------------------------
# 3. statistics
# ----------------------------------------------------------------------------------------------------------------
def ols_hc3(y, X, names, check_loo=True):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    assert h.max() < 1 - 1e-8, "an observation has leverage 1; HC3 undefined"
    meat = (X * ((e / (1 - h)) ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv
    assert np.isfinite(beta).all() and np.isfinite(V).all()
    assert np.allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0])
    if check_loo:  # exact identity: HC3 = sum_i (b - b_(-i))(b - b_(-i))'
        D = np.array([beta - np.linalg.lstsq(np.delete(X, i, 0), np.delete(y, i), rcond=None)[0] for i in range(n)])
        assert np.allclose(D.T @ D, V, rtol=1e-6, atol=1e-9), "HC3 self-check failed"
    se = np.sqrt(np.diag(V))
    df = n - k
    tcrit = stats.t.ppf(0.975, df)
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), df)
    ss_res, ss_tot = float(e @ e), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    coefs = {nm: {"coef": float(b), "se_hc3": float(s), "ci95": [float(b - tcrit * s), float(b + tcrit * s)],
                  "t": float(t), "p": float(pp)} for nm, b, s, t, pp in zip(names, beta, se, tval, p)}
    return {"n": int(n), "k": int(k), "df_resid": int(df), "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / df,
            "coefs": coefs}


def domain_dummies(dom):
    return np.column_stack([(dom == d).astype(float) for d in DOMAIN_ORDER[1:]]), [f"domain[{d}]" for d in DOMAIN_ORDER[1:]]


def fit(y, dom, preds, with_fe=True):
    """preds: list of (name, vector)."""
    cols, names = [np.ones_like(y)], ["const"]
    for nm, v in preds:
        cols.append(v)
        names.append(nm)
    if with_fe:
        D, dn = domain_dummies(dom)
        cols += list(D.T)
        names += dn
    return ols_hc3(y, np.column_stack(cols), names)


def zscore(v):
    return (v - v.mean()) / v.std(ddof=1)


def demean(v, dom):
    out = v.astype(float).copy()
    for d in np.unique(dom):
        out[dom == d] -= v[dom == d].mean()
    return out


def spearman(x, y):
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": int(len(x))}


def within_domain_spearman(x, y, dom, rng):
    """Spearman on domain-demeaned x and y; p-value also by permuting y within domain (keeps domain means fixed)."""
    xd, yd = demean(x, dom), demean(y, dom)
    r0 = stats.spearmanr(xd, yd)[0]
    idx = [np.where(dom == d)[0] for d in np.unique(dom)]
    hits = 0
    for _ in range(N_PERM):
        yp = yd.copy()
        for ix in idx:
            yp[ix] = yd[rng.permutation(ix)]
        hits += abs(stats.spearmanr(xd, yp)[0]) >= abs(r0) - 1e-12
    out = spearman(xd, yd)
    out["p_perm_within_domain"] = (hits + 1) / (N_PERM + 1)
    return out


def partial_spearman(x, y, covs):
    """Rank x, y and continuous covariates (pooled), residualize on [1, covs] (dummies kept as 0/1), Pearson r of
    residuals; t test with df = n - 2 - #covariates (as in pingouin.partial_corr(method='spearman'))."""
    n = len(x)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([np.ones(n)] + [c if is_dummy else stats.rankdata(c) for c, is_dummy in covs])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
    r = float(np.corrcoef(res(rx), res(ry))[0, 1])
    df = n - 2 - (Z.shape[1] - 1)
    t = r * np.sqrt(df / (1 - r * r))
    return {"rho_partial": r, "p": float(2 * stats.t.sf(abs(t), df)), "n": int(n), "df": int(df)}


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="paper/error_analysis/category_gain_controls.json")
    ap.add_argument("--stored", default="paper/error_analysis/category_analysis.json")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = {"meta": {"script": "scripts/category_gain_controls.py", "min_test": MIN_TEST, "rare_max_train": RARE_MAX,
                    "n_perm": N_PERM, "seed": SEED, "reference_domain": DOMAIN_ORDER[0],
                    "ci": "95% CI from HC3 SE with t(n-k) quantile (more conservative than the normal quantile)",
                    "similarity_units": "similarity and overlap enter regressions as z-scores over the analysed "
                                        "categories (coef = F1 points per 1 SD); log support in natural-log units"}}

    # ---- 1. rows + reproduction
    rows = build_rows()
    out["reproduction"] = {"row_check": check_against_stored(rows, ROOT / a.stored)}
    g = np.array([r["gain"] for r in rows])
    ls = np.log(np.array([r["train"] for r in rows], dtype=float))
    ov = np.array([r["overlap"] for r in rows])
    dom = np.array([r["domain"] for r in rows])
    rep = {"gain_logtrain": spearman(ls, g), "gain_overlap": spearman(ov, g), "overlap_logtrain": spearman(ov, ls)}
    rep["reported_in_paper"] = REPORTED
    rep["matches_reported_to_2dp"] = bool(
        len(rows) == REPORTED["n"]
        and all(round(rep[k]["rho"], 2) == REPORTED[k] for k in ["gain_logtrain", "gain_overlap", "overlap_logtrain"]))
    out["reproduction"]["spearman"] = rep
    out["n_categories_per_domain"] = {d: int((dom == d).sum()) for d in DOMAIN_ORDER}
    out["n_rare_per_domain"] = {d: int(((dom == d) & (np.exp(ls) <= RARE_MAX)).sum()) for d in DOMAIN_ORDER}
    print(f"n={len(rows)}  per domain {out['n_categories_per_domain']}  rare(<= {RARE_MAX}) {out['n_rare_per_domain']}")
    for k in ["gain_logtrain", "gain_overlap", "overlap_logtrain"]:
        print(f"  reproduce {k:17s} rho={rep[k]['rho']:+.3f} p={rep[k]['p']:.4f}  (paper {REPORTED[k]:+.2f})")
    print("  row check:", out["reproduction"]["row_check"])

    # ---- 2. similarity measures
    texts, identical = load_category_texts()
    out["category_texts"] = {"source": "outputs/cage_<domain>_fixed/seed_*/category_texts.json",
                             "identical_across_seeds": identical,
                             "n_categories_in_domain": {d: len(t) for d, t in texts.items()}}
    for d in DOMAIN_ORDER:  # every analysed category must have a description
        assert all(r["category"] in texts[d] for r in rows if r["domain"] == d), d
    sims, enc_info = {}, {}
    for m in MEASURES:
        if m == "jaccard":
            sims[m] = max_other_similarity(texts)
            enc_info[m] = {"model": None, "input": "PyVi word tokens, lower-cased, punctuation dropped"}
        else:
            emb, info = embed(texts, m)
            sims[m], enc_info[m] = max_other_similarity(texts, emb), info
            if m in CENTERED:  # anisotropy check: subtract the mean embedding of all descriptions (all domains)
                mu = np.vstack([E for _, E in emb.values()]).mean(0)
                sims[f"{m}_centered"] = max_other_similarity(texts, {d: (c, E - mu) for d, (c, E) in emb.items()})
        for r in rows:
            r[f"sim_{m}"], r[f"nearest_{m}"] = sims[m][(r["domain"], r["category"])]
            if f"{m}_centered" in sims:
                r[f"sim_{m}_centered"] = sims[f"{m}_centered"][(r["domain"], r["category"])][0]
    out["similarity_measures"] = enc_info
    S = {m: np.array([r[f"sim_{m}"] for r in rows]) for m in MEASURES}

    props = {}
    for m in MEASURES:
        s = S[m]
        props[m] = {
            "summary": {"mean": float(s.mean()), "sd": float(s.std(ddof=1)), "min": float(s.min()), "max": float(s.max())},
            "domain_means": {d: float(s[dom == d].mean()) for d in DOMAIN_ORDER},
            "pooled_spearman_with_overlap": spearman(s, ov), "pooled_spearman_with_logtrain": spearman(s, ls),
            "within_domain_spearman_with_overlap": spearman(demean(s, dom), demean(ov, dom)),
            "within_domain_spearman_with_logtrain": spearman(demean(s, dom), demean(ls, dom)),
            "nearest_neighbour_agrees_with_tfidf_overlap": float(np.mean(
                [r[f"nearest_{m}"] == r["overlap_nearest"] for r in rows])),
            "spearman_with_other_measures": {m2: float(stats.spearmanr(s, S[m2])[0]) for m2 in MEASURES if m2 != m},
        }
    out["similarity_properties"] = props
    ex = {m: {c: sims[m][("hotel", c)] for c in ["ROOMS#CLEANLINESS", "ROOM_AMENITIES#CLEANLINESS"]} for m in MEASURES}
    out["paper_example_pair"] = ex
    print("\nsimilarity measures (pooled Spearman with overlap / log support; within-domain versions)")
    for m in MEASURES:
        p = props[m]
        print(f"  {m:13s} mean={p['summary']['mean']:.3f} sd={p['summary']['sd']:.3f} | ov {p['pooled_spearman_with_overlap']['rho']:+.2f}"
              f" logtr {p['pooled_spearman_with_logtrain']['rho']:+.2f} | within: ov {p['within_domain_spearman_with_overlap']['rho']:+.2f}"
              f" logtr {p['within_domain_spearman_with_logtrain']['rho']:+.2f} | NN=tfidf {p['nearest_neighbour_agrees_with_tfidf_overlap']:.2f}")
    print("  paper example:", {m: {k.split('#')[0]: (round(v[0], 3), v[1]) for k, v in ex[m].items()} for m in MEASURES})

    # ---- 3. regressions (gain in F1 points)
    zov = zscore(ov)
    reg = {}
    reg["M0_domain_only"] = fit(g, dom, [])
    reg["M1_logsupport"] = fit(g, dom, [("log_support", ls)])
    reg["M1_no_domain_FE"] = fit(g, dom, [("log_support", ls)], with_fe=False)
    rare = (np.exp(ls) <= RARE_MAX).astype(float)
    reg["M1b_rare_indicator"] = fit(g, dom, [(f"rare(train<={RARE_MAX})", rare)])
    reg["M2_logsupport_overlap"] = fit(g, dom, [("log_support", ls), ("z_overlap", zov)])
    for m in MEASURES:
        zs = zscore(S[m])
        reg[f"M3a_logsupport_sim[{m}]"] = fit(g, dom, [("log_support", ls), (f"z_sim_{m}", zs)])
        reg[f"M3b_logsupport_overlap_sim[{m}]"] = fit(g, dom, [("log_support", ls), ("z_overlap", zov), (f"z_sim_{m}", zs)])
        cls = ls - ls.mean()
        reg[f"M4_interaction_EXPLORATORY[{m}]"] = fit(
            g, dom, [("log_support_centered", cls), (f"z_sim_{m}", zs), (f"log_support_centered:z_sim_{m}", cls * zs)])
    loo = {}
    for d in DOMAIN_ORDER:  # does the within-domain rarity slope survive dropping each domain?
        keep = dom != d
        dd = [x for x in DOMAIN_ORDER if x != d]
        X = np.column_stack([np.ones(keep.sum()), ls[keep]] + [(dom[keep] == x).astype(float) for x in dd[1:]])
        loo[f"drop_{d}"] = ols_hc3(g[keep], X, ["const", "log_support"] + [f"domain[{x}]" for x in dd[1:]])["coefs"]["log_support"]
        loo[f"drop_{d}"]["n"] = int(keep.sum())
    reg["M1_leave_one_domain_out_log_support"] = loo
    out["regressions"] = reg

    def show(name, keys=None):
        r = reg[name]
        print(f"  {name}  (n={r['n']}, R2={r['r2']:.3f}, adjR2={r['adj_r2']:.3f})")
        for k, c in r["coefs"].items():
            if k == "const" or k.startswith("domain[") and keys != "all":
                continue
            print(f"      {k:40s} {c['coef']:+7.3f}  [{c['ci95'][0]:+7.3f}, {c['ci95'][1]:+7.3f}]  p={c['p']:.4f}")

    print("\nOLS, domain FE, HC3")
    show("M0_domain_only")
    for nm in ["M1_no_domain_FE", "M1_logsupport", "M1b_rare_indicator", "M2_logsupport_overlap"]:
        show(nm)
    for m in MEASURES:
        show(f"M3a_logsupport_sim[{m}]")
        show(f"M3b_logsupport_overlap_sim[{m}]")
    for m in MEASURES:
        show(f"M4_interaction_EXPLORATORY[{m}]")
    print("  leave-one-domain-out log_support coef:",
          {k: (round(v["coef"], 2), round(v["p"], 4), v["n"]) for k, v in loo.items()})

    # ---- 4. rank-based checks
    rk = {"within_domain_spearman": {}, "partial_spearman": {}, "per_domain_spearman": {}, "rare_subset_EXPLORATORY": {}}
    W = rk["within_domain_spearman"]
    W["gain~log_support"] = within_domain_spearman(ls, g, dom, rng)
    W["gain~overlap"] = within_domain_spearman(ov, g, dom, rng)
    for m in MEASURES:
        W[f"gain~sim_{m}"] = within_domain_spearman(S[m], g, dom, rng)
    D, _ = domain_dummies(dom)
    dcov = [(c, True) for c in D.T]
    P = rk["partial_spearman"]
    P["gain~log_support | domain"] = partial_spearman(ls, g, dcov)
    P["gain~overlap | log_support"] = partial_spearman(ov, g, [(ls, False)])
    P["gain~overlap | log_support+domain"] = partial_spearman(ov, g, [(ls, False)] + dcov)
    for m in MEASURES:
        P[f"gain~sim_{m} | log_support"] = partial_spearman(S[m], g, [(ls, False)])
        P[f"gain~sim_{m} | log_support+domain"] = partial_spearman(S[m], g, [(ls, False)] + dcov)
        P[f"gain~sim_{m} | log_support+overlap+domain"] = partial_spearman(S[m], g, [(ls, False), (ov, False)] + dcov)
    for d in DOMAIN_ORDER:
        k = dom == d
        rk["per_domain_spearman"][d] = {"n": int(k.sum()), "gain~log_support": spearman(ls[k], g[k])["rho"],
                                        **{f"gain~sim_{m}": spearman(S[m][k], g[k])["rho"] for m in MEASURES}}
    rmask = np.exp(ls) <= RARE_MAX
    hmask = rmask & (dom == "hotel")
    R = rk["rare_subset_EXPLORATORY"]
    R["n"], R["n_hotel"] = int(rmask.sum()), int(hmask.sum())
    R["mean_gain_rare"], R["mean_gain_nonrare"] = float(g[rmask].mean()), float(g[~rmask].mean())
    for m in MEASURES:
        R[f"gain~sim_{m}"] = spearman(S[m][rmask], g[rmask])
        R[f"gain~sim_{m} (hotel rare only)"] = spearman(S[m][hmask], g[hmask])
    # pooled tertiles of domain-demeaned primary similarity (descriptive, mirrors the paper's overlap tertiles)
    sd_ = demean(S[PRIMARY], dom)
    q = np.quantile(sd_, [1 / 3, 2 / 3])
    rk["tertiles_domain_demeaned_" + PRIMARY] = {
        lab: {"n": int(k.sum()), "mean_gain": float(g[k].mean()), "median_gain": float(np.median(g[k])),
              "median_train": float(np.median(np.exp(ls[k])))}
        for lab, k in [("low", sd_ < q[0]), ("mid", (sd_ >= q[0]) & (sd_ < q[1])), ("high", sd_ >= q[1])]}
    out["rank_based"] = rk

    print("\nwithin-domain (demeaned) Spearman  [p, within-domain permutation p]")
    for k, v in W.items():
        print(f"  {k:24s} rho={v['rho']:+.3f}  p={v['p']:.4f}  p_perm={v['p_perm_within_domain']:.4f}")
    print("partial Spearman")
    for k, v in P.items():
        print(f"  {k:48s} rho={v['rho_partial']:+.3f}  p={v['p']:.4f}  df={v['df']}")
    print("per-domain Spearman (gain~log support, gain~sim_*):")
    for d, v in rk["per_domain_spearman"].items():
        print(f"  {d:10s} n={v['n']:2d} " + "  ".join(f"{k.split('~')[1]}={x:+.2f}" for k, x in v.items() if k != "n"))
    print(f"rare subset (train<={RARE_MAX}) n={R['n']} (hotel {R['n_hotel']}): "
          + "  ".join(f"{m}={R[f'gain~sim_{m}']['rho']:+.2f}(p={R[f'gain~sim_{m}']['p']:.2f})" for m in MEASURES))
    print("  hotel-rare only: " + "  ".join(f"{m}={R[f'gain~sim_{m} (hotel rare only)']['rho']:+.2f}" for m in MEASURES))
    print("tertiles of domain-demeaned", PRIMARY, rk["tertiles_domain_demeaned_" + PRIMARY])

    # ---- 5. robustness of M1 / M3a (none of these subsets select on the outcome directly)
    rob = {}
    dcov_of = lambda dm: [(c, True) for c in domain_dummies(dm)[0].T]  # noqa: E731

    def m1_m3a(mask, y=None, sims_=MEASURES):
        y = g if y is None else y
        res = {"n": int(mask.sum()), "M1_log_support": fit(y[mask], dom[mask], [("log_support", ls[mask])])["coefs"]["log_support"]}
        for m in sims_:
            s = np.array([r[f"sim_{m}"] for r in rows])
            res[f"M3a_z_sim[{m}]"] = fit(y[mask], dom[mask], [("log_support", ls[mask]), ("z", zscore(s[mask]))])["coefs"]["z"]
            res[f"partial_spearman[{m}] | log_support+domain"] = partial_spearman(
                s[mask], y[mask], [(ls[mask], False)] + dcov_of(dom[mask]))
        return res

    allm = np.ones(len(rows), bool)
    rob["min_test_10"] = m1_m3a(np.array([r["test"] for r in rows]) >= 10)  # category_analysis.py's default threshold
    rob["rank_of_gain_OLS"] = m1_m3a(allm, y=stats.rankdata(g))             # outcome = rank of gain (outlier-robust)
    rob["rank_of_gain_OLS"]["note"] = "coefficients in rank units (1..74) per ln-support / per SD of similarity"
    X1 = np.column_stack([np.ones(len(g)), ls, domain_dummies(dom)[0]])     # Cook's distance from M1
    b1 = np.linalg.lstsq(X1, g, rcond=None)[0]
    e1 = g - X1 @ b1
    h1 = np.einsum("ij,jk,ik->i", X1, np.linalg.inv(X1.T @ X1), X1)
    cook = e1 ** 2 * h1 / (X1.shape[1] * (e1 @ e1 / (len(g) - X1.shape[1])) * (1 - h1) ** 2)
    keep = cook <= 4 / len(g)
    rob["exclude_cooks_d_gt_4_over_n"] = m1_m3a(keep)
    rob["exclude_cooks_d_gt_4_over_n"]["excluded"] = [
        f"{r['domain']}/{r['category']} (gain {r['gain']:+.1f}, test {r['test']})" for r, k in zip(rows, keep) if not k]
    rob["exclude_cooks_d_gt_4_over_n"]["within_domain_spearman_gain~log_support"] = spearman(
        demean(ls[keep], dom[keep]), demean(g[keep], dom[keep]))
    rob["centered_embeddings"] = m1_m3a(allm, sims_=[f"{m}_centered" for m in CENTERED])
    out["robustness"] = rob
    print("\nrobustness (M1 log_support coef; M3a z_sim coef; partial Spearman | log support + domain)")
    for nm, v in rob.items():
        c = v["M1_log_support"]
        print(f"  {nm:28s} n={v['n']}  M1 log_support {c['coef']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] p={c['p']:.4f}")
        for k, c in v.items():
            if k.startswith("M3a"):
                ps = v[k.replace("M3a_z_sim", "partial_spearman") + " | log_support+domain"]
                print(f"      {k:34s} {c['coef']:+6.2f} [{c['ci95'][0]:+6.2f}, {c['ci95'][1]:+6.2f}] p={c['p']:.3f}"
                      f" | partial rho {ps['rho_partial']:+.3f} p={ps['p']:.3f}")
        if "excluded" in v:
            print("      excluded:", v["excluded"], "| within-domain rho(gain, log support) after exclusion:",
                  round(v["within_domain_spearman_gain~log_support"]["rho"], 3),
                  "p=%.4f" % v["within_domain_spearman_gain~log_support"]["p"])

    out["rows"] = rows
    (ROOT / a.json).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nwrote", a.json)


if __name__ == "__main__":
    main()
