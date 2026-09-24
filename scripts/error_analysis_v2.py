#!/usr/bin/env python3
"""Redo Section 6 (Case Study and Error Analysis) on the new 5-seed CAGE (fixed fusion, description)
runs, seed 42 (matched to the seed used for sampling, as in the original protocol).

1. error population: every (review, category) where CAGE's predicted label != gold label
2. error profile table: per domain, missed / spurious / polarity counts + share also wrong in Ensemble BERTs
3. sample 20 errors per domain (seed 42, deterministic) for manual coding
"""
import json, random
from pathlib import Path

DOMAINS = ["restaurant", "hotel", "phone", "education", "beauty"]
ROOT = Path("outputs")

def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        gold = {g["category"]: g["sentiment"] for g in r["gold"]}
        pred = {p["category"]: p["sentiment"] for p in r["prediction"]}
        rows[r["id"]] = {"text": r["text"], "gold": gold, "pred": pred}
    return rows

def errors_for(domain):
    cage = load(ROOT / f"cage_{domain}_fixed/seed_42/test_predictions.jsonl")
    ens = load(ROOT / f"bert_{domain}_ensemble/test_predictions.jsonl")
    all_cats = set()
    for r in cage.values():
        all_cats |= set(r["gold"]) | set(r["pred"])
    errs = []
    for rid, r in cage.items():
        for cat in sorted(all_cats):
            g = r["gold"].get(cat, "none"); p = r["pred"].get(cat, "none")
            if g == p:
                continue
            if g == "none":
                etype = "spurious"
            elif p == "none":
                etype = "missed"
            else:
                etype = "polarity"
            e_pred = ens.get(rid, {}).get("pred", {}).get(cat, "none") if rid in ens else None
            e_row = ens.get(rid)
            e_pred = e_row["pred"].get(cat, "none") if e_row else None
            shared = (e_pred is not None) and (e_pred != g)
            errs.append({"domain": domain, "id": rid, "text": r["text"], "category": cat,
                         "gold": g, "cage": p, "ensemble": e_pred, "type": etype, "shared": shared})
    return errs

def confusion(errs, etype):
    from collections import Counter
    c = Counter((e["gold"], e["cage"]) for e in errs if e["type"] == etype)
    return c.most_common(5)

def main():
    profile = {}
    all_errs = {}
    for d in DOMAINS:
        errs = errors_for(d)
        all_errs[d] = errs
        n = len(errs); missed = sum(e["type"] == "missed" for e in errs)
        spurious = sum(e["type"] == "spurious" for e in errs)
        polarity = sum(e["type"] == "polarity" for e in errs)
        shared = sum(e["shared"] for e in errs if e["ensemble"] is not None)
        shared_n = sum(1 for e in errs if e["ensemble"] is not None)
        profile[d] = {"n": n, "missed": missed, "spurious": spurious, "polarity": polarity,
                      "shared_pct": round(100 * shared / shared_n, 1) if shared_n else None, "shared_n": shared_n}
        print(f"{d:10s} n={n:5d} missed={missed:5d} spurious={spurious:5d} polarity={polarity:5d} "
              f"shared={profile[d]['shared_pct']}% (of {shared_n})")
        print("  top polarity confusions:", confusion(errs, "polarity"))
    json.dump(profile, open("paper/error_analysis/profile_v2.json", "w"), indent=2, ensure_ascii=False)

    # sample 20 errors per domain, seed 42, deterministic order (sort by id,category first)
    sample = {}
    for d in DOMAINS:
        errs = sorted(all_errs[d], key=lambda e: (e["id"], e["category"]))
        rng = random.Random(42)
        sample[d] = rng.sample(errs, min(20, len(errs)))
    json.dump(sample, open("paper/error_analysis/sample_v2.json", "w"), indent=2, ensure_ascii=False)
    print("\nwrote paper/error_analysis/profile_v2.json and sample_v2.json")
    tot = sum(p["n"] for p in profile.values())
    print("TOTAL errors:", tot)

if __name__ == "__main__":
    main()
