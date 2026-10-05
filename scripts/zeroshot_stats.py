"""Summarise the leave-one-category-out zero-shot ablation (run_cage_zeroshot.sh).

For each (domain, category) combo, compares that category's own F1 (joint category+polarity pair,
same metric as scripts/category_analysis.py's per_category_f1) under two conditions:
  zero-shot   outputs/cage_zeroshot_<domain>_<slug>/seed_{42,123,2024}/test_predictions.jsonl
              (the category's training labels were withheld entirely; dev/test kept real gold)
  supervised  outputs/cage_<domain>_fixed/seed_{42,123,2024,7,99}/test_predictions.jsonl
              (the already-completed Table 4 main run -- the category was trained normally)

Zero-shot has 3 seeds, supervised has 5 -- different seed counts, so this is an unpaired (Welch,
two-sample, one-sided: supervised > zero-shot) comparison, the same test cage_stats.py's
ensemble_significance() uses for two independently-trained distributions.

Usage: python3 scripts/zeroshot_stats.py [--out_dir outputs] [--json paper/error_analysis/zeroshot_stats.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

# Mirrors the TABLE in run_cage_zeroshot.sh -- keep in sync if that table changes.
TABLE = [
    ("restaurant", "most", "FOOD#QUALITY", "food_quality"),
    ("restaurant", "median", "RESTAURANT#MISCELLANEOUS", "restaurant_miscellaneous"),
    ("restaurant", "rarest", "DRINKS#PRICES", "drinks_prices"),
    ("hotel", "most", "SERVICE#GENERAL", "service_general"),
    ("hotel", "median", "FOOD&DRINKS#STYLE&OPTIONS", "food_drinks_style_options"),
    ("hotel", "rarest", "FACILITIES#CLEANLINESS", "facilities_cleanliness"),
    ("phone", "most", "GENERAL", "general"),
    ("phone", "median", "PRICE", "price"),
    ("phone", "rarest", "STORAGE", "storage"),
    ("education", "most", "Hành vi", "hanh_vi"),
    ("education", "median", "Nói chung", "noi_chung"),
    ("education", "rarest", "Cung cấp tài liệu", "cung_cap_tai_lieu"),
    ("beauty", "most", "Màu sắc", "mau_sac"),
    ("beauty", "median", "Bao bì", "bao_bi"),
    ("beauty", "rarest", "Độ bền màu", "do_ben_mau"),
]
DISPLAY = {"restaurant": "Restaurant", "hotel": "Hotel", "phone": "Phone", "education": "Education", "beauty": "Beauty"}


def category_f1_by_seed(base: Path, category: str):
    """{seed: F1%} for one category, restricted to test_predictions.jsonl files under base/seed_*/."""
    res = {}
    for f in base.glob("seed_*/test_predictions.jsonl"):
        seed = int(f.parent.name.split("_")[1])
        tp = fp = fn = 0
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            g = {x["category"]: x["sentiment"] for x in r["gold"]}
            p = {x["category"]: x["sentiment"] for x in r["prediction"]}
            if category not in g and category not in p:
                continue
            ok = category in g and category in p and g[category] == p[category]
            tp += ok
            fp += category in p and not ok
            fn += category in g and not ok
        res[seed] = 200 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return res


def ms(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--json", default="paper/error_analysis/zeroshot_stats.json")
    ap.add_argument("--latex", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    R = {}

    print("Zero-shot leave-one-category-out vs supervised (same category, joint category+polarity F1, %)")
    print(f"{'domain':11s}{'tier':8s}{'category':28s}{'zero-shot':>14s}{'supervised':>14s}{'gap':>8s}{'p':>8s}  n_zs/n_sup")
    for domain, tier, category, slug in TABLE:
        zs_dir = out_dir / f"cage_zeroshot_{domain}_{slug}"
        sup_dir = out_dir / f"cage_{domain}_fixed"
        zs = category_f1_by_seed(zs_dir, category)
        sup = category_f1_by_seed(sup_dir, category)
        if not zs or not sup:
            print(f"{domain:11s}{tier:8s}{category:28s}  (missing: zero-shot n={len(zs)}, supervised n={len(sup)})")
            continue
        (zm, zsd), (sm, ssd) = ms(list(zs.values())), ms(list(sup.values()))
        if len(zs) > 1 and len(sup) > 1:
            t, p = stats.ttest_ind(list(sup.values()), list(zs.values()), equal_var=False, alternative="greater")
        else:
            t, p = float("nan"), float("nan")
        key = f"{domain}/{tier}/{category}"
        R[key] = {"category": category, "tier": tier, "slug": slug,
                  "zeroshot_mean": zm, "zeroshot_std": zsd, "zeroshot_n": len(zs), "zeroshot_values": list(zs.values()),
                  "supervised_mean": sm, "supervised_std": ssd, "supervised_n": len(sup), "supervised_values": list(sup.values()),
                  "gap": sm - zm, "welch_t": float(t), "p_one_sided": float(p)}
        print(f"{domain:11s}{tier:8s}{category:28s}{zm:8.2f}±{zsd:.2f}  {sm:8.2f}±{ssd:.2f}  {sm-zm:+8.2f}{p:8.4f}  {len(zs)}/{len(sup)}")

    print("\nBy tier (mean gap across domains, only combos where both sides are present)")
    for tier in ("most", "median", "rarest"):
        gaps = [v["gap"] for k, v in R.items() if v.get("tier") == tier]  # .get: R also holds "by_tier/*"
        # summary rows added by this same loop (no "tier" key), which R.items() sees once we mutate R below.
        if gaps:
            print(f"  {tier:8s} n={len(gaps)}  mean gap {np.mean(gaps):+.2f}  median {np.median(gaps):+.2f}")
            R[f"by_tier/{tier}"] = {"n": len(gaps), "mean_gap": float(np.mean(gaps)), "median_gap": float(np.median(gaps))}

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))
    print("\nwrote", a.json)

    if a.latex:
        L = Path(a.latex); L.mkdir(parents=True, exist_ok=True)
        rows = []
        for domain, tier, category, slug in TABLE:
            v = R.get(f"{domain}/{tier}/{category}")
            if v:
                rows.append(f"{DISPLAY[domain]} & {tier.capitalize()} & {category} & "
                            f"${v['supervised_mean']:.2f}\\pm{v['supervised_std']:.2f}$ & "
                            f"${v['zeroshot_mean']:.2f}\\pm{v['zeroshot_std']:.2f}$ & "
                            f"${v['gap']:+.2f}$ & {v['p_one_sided']:.3f} \\\\")
        (L / "tab_zeroshot.tex").write_text("\n".join(rows) + "\n")
        print("wrote LaTeX rows to", L)


if __name__ == "__main__":
    main()
