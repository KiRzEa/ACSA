"""Aggregate counterfactual query-substitution runs (run_cage_queryswap.sh -> outputs/cage_qs_<domain>_fixed/seed_*/query_swap.json).

Arms re-evaluate every slot c with a substituted query: own (reference), nearest / farthest other category in the model's
own query space, a random other category, the mean of all queries, or zero.  Reports, per domain, micro-F1 (%) of each arm
(mean +- std over seeds), and a dose-response check: across categories, Spearman correlation between cos(q_c, q_nearest(c))
and the F1 that slot c keeps when it is answered with its nearest neighbour's query (per-category F1 averaged over seeds).
Because the heads are shared and queries do not interact, substituting q_j for q_c reproduces category j's decision at slot c;
the F1 against category c's gold is therefore a measure of how much c's decision is determined by (and distinct from) its query.
Usage: python3 scripts/query_swap_stats.py [--prefix cage_qs] [--out_dir outputs] [--json paper/error_analysis/query_swap_stats.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

DOMAINS = ["restaurant", "hotel", "phone", "education", "beauty"]
ARMS = ["own", "nearest", "farthest", "random", "mean", "zero"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--prefix", default="cage_qs")
    ap.add_argument("--json", default="paper/error_analysis/query_swap_stats.json")
    a = ap.parse_args()
    R, pooled_x, pooled_y = {}, [], []
    print(f"{'domain':11s}{'n':>3s}" + "".join(f"{x:>14s}" for x in ARMS) + "   dose-response (cos vs F1 with nearest query)")
    for d in DOMAINS:
        files = sorted(Path(a.out_dir, f"{a.prefix}_{d}_fixed").glob("seed_*/query_swap.json"))
        if not files:
            continue
        runs = [json.loads(f.read_text()) for f in files]
        cells = []
        for arm in ARMS:
            v = np.array([100 * r["arms"][arm]["acsa_f1_micro"] for r in runs])
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            R[f"{d}/{arm}"] = {"mean": float(v.mean()), "std": float(sd), "values": v.tolist()}
            cells.append(f"{v.mean():7.2f}±{sd:4.2f}")
        cos = np.mean([r["cos_nearest"] for r in runs], axis=0)
        f1n = np.mean([r["arms"]["nearest"]["per_category_f1"] for r in runs], axis=0)
        gold = np.mean([r["arms"]["own"]["gold_pairs"] for r in runs], axis=0)
        keep = gold >= 5
        rho, p = stats.spearmanr(cos[keep], f1n[keep]) if keep.sum() >= 4 else (float("nan"), float("nan"))
        R[f"{d}/dose_response"] = {"spearman": float(rho), "p": float(p), "n_categories": int(keep.sum())}
        pooled_x += cos[keep].tolist(); pooled_y += f1n[keep].tolist()
        print(f"{d:11s}{len(runs):3d}" + "".join(f"{c:>14s}" for c in cells) + f"   rho={rho:+.2f} (p={p:.3f}, n={int(keep.sum())})")
    if len(pooled_x) >= 4:
        rho, p = stats.spearmanr(pooled_x, pooled_y)
        R["pooled/dose_response"] = {"spearman": float(rho), "p": float(p), "n_categories": len(pooled_x)}
        print(f"{'pooled':11s}{'':3s}{'':84s}   rho={rho:+.2f} (p={p:.3f}, n={len(pooled_x)})")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(R, indent=2, default=float))
    print("wrote", a.json)


if __name__ == "__main__":
    main()
