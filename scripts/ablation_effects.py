"""Effect sizes, 95% CIs and multiple-testing correction for CAGE's paired ablation comparisons.

Answers reviewer Major Comment 3: every design-choice comparison in the paper is reported with its mean paired
difference, the 95% t-interval of that difference, Cohen's d_z, the raw two-sided paired t-test p, and p-values
adjusted for multiplicity (Holm and Bonferroni) within a declared correction set.

Scoring (per run): set-based (category, sentiment) pair micro-F1 from seed_<s>/test_predictions.jsonl, the same
scorer as scripts/cage_stats.py::_pred_f1 (metrics.json is NOT used). Pairing is by seed over the seeds both runs
share (normally 42, 123, 2024, 7, 99).

Families (= one table in the paper each) and their sign convention (positive = first-named side is better):
  fusion     tab:fusion-compare     Learned - Fixed       cage_<d>_learned          vs cage_<d>_fixed        wins: learned > fixed
  namedesc   tab:namedesc           Description - Name    cage_<d>_fixed            vs cage_name_<d>_fixed   wins: desc > name
  design     tab:design-ablation    CAGE - ablated        cage_<d>_fixed            vs cage_abl_<a>_<d>_fixed wins: CAGE > ablated
  component  tab:component-ablation CAGE - ablated        (Restaurant only)                                    wins: CAGE > ablated
  conditioning (pre-specified)      CAGE - ablated        cond_none, cond_concat, cond_attn_only, minimal_joint x 5 domains

Correction sets: Holm/Bonferroni are applied within a correction set, never across sets.
  design_choices_39  = fusion + namedesc + design + component (the 39 comparisons the paper reports; exploratory,
                       i.e. not specified before the results were seen)
  conditioning       = the pre-specified conditioning ablations, own Holm correction (confirmatory). Runs that do
                       not exist yet are skipped and listed under "skipped"; m stays the planned size (missing
                       comparisons count as p = 1), so a partial family is never under-corrected.
p_holm_family additionally adjusts within each paper table (a less conservative, per-table view).

Usage: python3 scripts/ablation_effects.py [--out_dir outputs] [--json paper/error_analysis/ablation_effects.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

DOMAINS = ["restaurant", "hotel", "phone", "education", "beauty"]
SEEDS = [42, 123, 2024, 7, 99]
COMPONENT_ABLS = ["lw_fixed", "acd_focal", "acd_asl", "child_tuning", "heads2", "heads4", "heads12", "heads16",
                  "heads24", "adapter64", "adapter96", "adapter256", "adapter384", "adapter512"]
DESIGN_ABLS = ["gate_hard", "gate_none", "query_id"]
CONDITIONING_ABLS = ["cond_none", "cond_concat", "cond_attn_only", "minimal_joint"]
LABEL = {"gate_hard": "Hard gate", "gate_none": "No gate", "query_id": "Free embedding",
         "lw_fixed": "Fixed loss weights", "acd_focal": "Focal loss", "acd_asl": "Asymmetric loss",
         "child_tuning": "Child-Tuning", "cond_none": "No conditioning", "cond_concat": "Concat conditioning",
         "cond_attn_only": "Attention-only conditioning", "minimal_joint": "Minimal joint"}
for _h in (2, 4, 12, 16, 24): LABEL[f"heads{_h}"] = f"{_h} heads"
for _a in (64, 96, 256, 384, 512): LABEL[f"adapter{_a}"] = f"Adapter {_a}"


def comparisons():
    """(family, correction_set, status, name, domain, a_dir, b_dir, contrast) for every planned comparison.
    The difference is always a - b."""
    out = []
    for d in DOMAINS:
        out.append(("fusion", "design_choices_39", "exploratory", "Learned fusion", d,
                    f"cage_{d}_learned", f"cage_{d}_fixed", "Learned - Fixed"))
    for d in DOMAINS:
        out.append(("namedesc", "design_choices_39", "exploratory", "Description vs name", d,
                    f"cage_{d}_fixed", f"cage_name_{d}_fixed", "Description - Name"))
    for abl in DESIGN_ABLS:
        for d in DOMAINS:
            out.append(("design", "design_choices_39", "exploratory", LABEL[abl], d,
                        f"cage_{d}_fixed", f"cage_abl_{abl}_{d}_fixed", "CAGE - ablated"))
    for abl in COMPONENT_ABLS:
        out.append(("component", "design_choices_39", "exploratory", LABEL[abl], "restaurant",
                    "cage_restaurant_fixed", f"cage_abl_{abl}_restaurant_fixed", "CAGE - ablated"))
    for abl in CONDITIONING_ABLS:
        for d in DOMAINS:
            out.append(("conditioning", "conditioning", "confirmatory (pre-specified)", LABEL[abl], d,
                        f"cage_{d}_fixed", f"cage_abl_{abl}_{d}_fixed", "CAGE - ablated"))
    return out


def pred_f1(path):
    tp = fp = fn = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g = {(x["category"], x["sentiment"]) for x in r["gold"]}
        p = {(x["category"], x["sentiment"]) for x in r["prediction"]}
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    return 200 * tp / (2 * tp + fp + fn) if tp else 0.0


_CACHE = {}


def load_run(out_dir, name):
    """{seed: test micro-F1 %} for outputs/<name>/seed_<s>/test_predictions.jsonl, {} if absent."""
    if name not in _CACHE:
        base, res = Path(out_dir) / name, {}
        for f in base.glob("seed_*/test_predictions.jsonl"):
            res[int(f.parent.name.split("_")[1])] = pred_f1(f)
        _CACHE[name] = res
    return _CACHE[name]


def paired_effect(a, b):
    seeds = [s for s in SEEDS if s in a and s in b] + sorted(set(a) & set(b) - set(SEEDS))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    p = float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue)
    av, bv = np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])
    return {"n": n, "seeds": seeds, "mean_diff": mean, "sd_diff": sd, "ci95": [mean - half, mean + half],
            "ci95_half_width": half, "d_z": mean / sd if sd > 0 else float("nan"), "p": p,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "a_mean": float(av.mean()), "a_sd": float(av.std(ddof=1)),
            "b_mean": float(bv.mean()), "b_sd": float(bv.std(ddof=1)), "diffs": d.tolist()}


def holm(p):
    """Holm step-down adjusted p-values (monotone, capped at 1), same order as input."""
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def bonferroni(p):
    return [min(1.0, len(p) * x) for x in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--json", default="paper/error_analysis/ablation_effects.json")
    a = ap.parse_args()

    rows, skipped = [], []
    for fam, cset, status, name, d, adir, bdir, contrast in comparisons():
        ra, rb = load_run(a.out_dir, adir), load_run(a.out_dir, bdir)
        shared = set(ra) & set(rb)
        if len(shared) < 2:
            skipped.append({"family": fam, "comparison": name, "domain": d, "a_dir": adir, "b_dir": bdir,
                            "reason": f"{len(ra)} seeds in a, {len(rb)} in b, {len(shared)} shared"})
            continue
        rows.append({"family": fam, "correction_set": cset, "status": status, "comparison": name, "domain": d,
                     "contrast": contrast, "a_dir": adir, "b_dir": bdir, **paired_effect(ra, rb)})

    # m is the PLANNED size of the set/family: a comparison whose runs are missing enters as p = 1, so a
    # partially finished pre-specified family is never corrected for fewer tests than it declared.
    planned = {}
    for c in comparisons():
        for key, g in (("correction_set", c[1]), ("family", c[0])): planned[key, g] = planned.get((key, g), 0) + 1
    for key, field in (("correction_set", "p_holm"), ("family", "p_holm_family")):
        for g in dict.fromkeys(r[key] for r in rows):
            idx = [i for i, r in enumerate(rows) if r[key] == g]
            ps = [rows[i]["p"] for i in idx] + [1.0] * (planned[key, g] - len(idx))
            for i, h in zip(idx, holm(ps)): rows[i][field] = h
            if key == "correction_set":
                for i, bo in zip(idx, bonferroni(ps)):
                    rows[i]["p_bonferroni"] = bo; rows[i]["m_correction_set"] = len(ps)

    summary = {}
    for cset in dict.fromkeys(r["correction_set"] for r in rows):
        rs = [r for r in rows if r["correction_set"] == cset]
        hw = np.array([r["ci95_half_width"] for r in rs])
        summary[cset] = {
            "m_planned": planned["correction_set", cset], "m_present": len(rs),
            "complete": len(rs) == planned["correction_set", cset], "n_raw_p_lt_05": sum(r["p"] < 0.05 for r in rs),
            "n_holm_lt_05": sum(r["p_holm"] < 0.05 for r in rs),
            "n_bonferroni_lt_05": sum(r["p_bonferroni"] < 0.05 for r in rs),
            "n_ci_excludes_zero": sum(r["ci95"][0] > 0 or r["ci95"][1] < 0 for r in rs),
            # how many 95% CIs are narrow enough to rule out an effect of this size in either direction
            "n_ci95_within_pm": {str(m): sum(-m < r["ci95"][0] and r["ci95"][1] < m for r in rs) for m in (0.5, 1.0)},
            "ci95_half_width": {"min": float(hw.min()), "median": float(np.median(hw)), "mean": float(hw.mean()),
                                "max": float(hw.max())},
            "sd_diff_median": float(np.median([r["sd_diff"] for r in rs])),
            "abs_mean_diff_max": float(max(abs(r["mean_diff"]) for r in rs))}
    for fam in dict.fromkeys(r["family"] for r in rows):
        rs = [r for r in rows if r["family"] == fam]
        summary[f"family/{fam}"] = {"m_planned": planned["family", fam], "m_present": len(rs), "n_holm_family_lt_05": sum(r["p_holm_family"] < 0.05 for r in rs),
                                    "ci95_half_width_median": float(np.median([r["ci95_half_width"] for r in rs]))}

    hdr = (f"{'family':10s} {'comparison':20s} {'domain':10s} {'n':>2s} {'diff':>6s} {'SD':>5s} {'95% CI':>16s} "
           f"{'d_z':>6s} {'wins':>4s} {'p':>6s} {'Holm':>6s} {'Bonf':>6s} {'Holm/tbl':>8s}")
    for cset in dict.fromkeys(r["correction_set"] for r in rows):
        rs = [r for r in rows if r["correction_set"] == cset]
        mp = planned["correction_set", cset]
        print(f"\n== correction set: {cset} ({rs[0]['status']}; m={mp}, {len(rs)} present"
              f"{'' if len(rs) == mp else ', missing runs count as p=1 -- INCOMPLETE'}) ==")
        print(hdr)
        for r in rs:
            flag = " *" if r["p"] < 0.05 else ""
            print(f"{r['family']:10s} {r['comparison'][:20]:20s} {r['domain']:10s} {r['n']:2d} {r['mean_diff']:+6.2f} "
                  f"{r['sd_diff']:5.2f} [{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}] {r['d_z']:+6.2f} "
                  f"{r['wins']}/{r['n']} {r['p']:6.3f} {r['p_holm']:6.3f} {r['p_bonferroni']:6.3f} "
                  f"{r['p_holm_family']:8.3f}{flag}")
        s = summary[cset]
        print(f"raw p<0.05: {s['n_raw_p_lt_05']}/{s['m_present']} | Holm<0.05: {s['n_holm_lt_05']} | "
              f"Bonferroni<0.05: {s['n_bonferroni_lt_05']} | CI half-width min/median/max "
              f"{s['ci95_half_width']['min']:.2f}/{s['ci95_half_width']['median']:.2f}/{s['ci95_half_width']['max']:.2f} F1 | "
              f"CI inside ±0.5 / ±1.0 F1: {s['n_ci95_within_pm']['0.5']} / {s['n_ci95_within_pm']['1.0']}")
    if skipped:
        print(f"\nskipped {len(skipped)} comparisons with missing runs: "
              + ", ".join(sorted({f"{s['family']}:{s['comparison']}" for s in skipped})))

    out = {"meta": {"scorer": "set-based (category, sentiment) pair micro-F1 from test_predictions.jsonl",
                    "pairing": "by seed, shared seeds", "ci": "95% t-interval of mean paired difference, df=n-1",
                    "effect_size": "Cohen's d_z = mean_diff / sd_diff (sd ddof=1)",
                    "test": "two-sided paired t-test (scipy.stats.ttest_rel)",
                    "p_holm": "Holm step-down within correction_set", "p_bonferroni": "Bonferroni within correction_set",
                    "p_holm_family": "Holm within family (one paper table)",
                    "wins": "seeds where a - b > 0 (a = first side of contrast)"},
           "summary": summary, "comparisons": rows, "skipped": skipped}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
