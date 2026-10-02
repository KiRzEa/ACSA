"""Aggregate the query-paraphrase-robustness eval (scripts/paraphrase_robustness.py ->
outputs/cage_ckpt_<domain>_fixed/seed_42/paraphrase_eval.json).

One checkpoint per domain (see run_cage_checkpoint.sh's docstring for why), so unlike cage_stats.py/
query_swap_stats.py there is no seed variance to report -- the statistical unit here is *category*:
for each domain, every paraphrase round's per-category F1 is paired against the "original" round's
same category (n = number of categories in that domain), two-sided paired t-test. This mirrors how
scripts/query_swap_stats.py's dose-response check already treats per-category F1s within a single
model as the sample population.

Usage: python3 scripts/paraphrase_stats.py [--out_dir outputs] [--json paper/error_analysis/paraphrase_stats.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

DOMAINS = ["restaurant", "hotel", "phone", "education", "beauty"]
DISPLAY = {"restaurant": "Restaurant", "hotel": "Hotel", "phone": "Phone", "education": "Education", "beauty": "Beauty"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--json", default="paper/error_analysis/paraphrase_stats.json")
    ap.add_argument("--latex", default=None)
    a = ap.parse_args()
    R = {}
    pooled_deltas = []

    for d in DOMAINS:
        f = Path(a.out_dir, f"cage_ckpt_{d}_fixed", "seed_42", "paraphrase_eval.json")
        if not f.exists():
            print(f"{d:11s} (missing: {f})")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        cats = data["categories"]
        orig = np.array(data["rounds"]["original"])
        para_rounds = sorted(k for k in data["rounds"] if k != "original")
        print(f"\n{DISPLAY[d]} (n={len(cats)} categories, original mean F1={orig.mean():.2f})")
        print(f"{'round':10s}{'mean F1':>10s}{'mean delta':>12s}{'categories worse':>18s}{'t':>8s}{'p (2-sided)':>12s}")
        for rname in para_rounds:
            vals = np.array(data["rounds"][rname])
            delta = vals - orig
            t, p = stats.ttest_1samp(delta, 0.0) if len(delta) > 1 else (float("nan"), float("nan"))
            worse = int((delta < 0).sum())
            R[f"{d}/{rname}"] = {"mean_f1": float(vals.mean()), "mean_delta": float(delta.mean()),
                                  "n_categories": len(cats), "n_worse": worse,
                                  "t": float(t), "p_two_sided": float(p),
                                  "per_category": {c: {"original": float(o), rname: float(v), "delta": float(v - o)}
                                                    for c, o, v in zip(cats, orig, vals)}}
            pooled_deltas += delta.tolist()
            print(f"{rname:10s}{vals.mean():10.2f}{delta.mean():+12.2f}{worse:11d}/{len(cats):<6d}{t:8.2f}{p:12.4f}")

    if pooled_deltas:
        pd = np.array(pooled_deltas)
        t, p = stats.ttest_1samp(pd, 0.0)
        R["pooled"] = {"n": len(pd), "mean_delta": float(pd.mean()), "std_delta": float(pd.std(ddof=1)),
                        "t": float(t), "p_two_sided": float(p)}
        print(f"\npooled across all domains/rounds: n={len(pd)}  mean delta={pd.mean():+.2f}±{pd.std(ddof=1):.2f}  p={p:.4f}")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))
    print("\nwrote", a.json)

    if a.latex:
        L = Path(a.latex); L.mkdir(parents=True, exist_ok=True)
        rows = []
        for d in DOMAINS:
            para_rounds = sorted({k.split("/")[1] for k in R if k.startswith(f"{d}/")})
            for rname in para_rounds:
                v = R.get(f"{d}/{rname}")
                if v:
                    rows.append(f"{DISPLAY[d]} & {rname.replace('_', ' ').capitalize()} & "
                                f"${v['mean_f1']:.2f}$ & ${v['mean_delta']:+.2f}$ & "
                                f"{v['n_worse']}/{v['n_categories']} & {v['p_two_sided']:.3f} \\\\")
        (L / "tab_paraphrase.tex").write_text("\n".join(rows) + "\n")
        print("wrote LaTeX rows to", L)


if __name__ == "__main__":
    main()
