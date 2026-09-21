"""Summarise CAGE 5-seed runs (mean +- std, stability vs single-run baselines, config selection, ablation).

Inputs (per domain d, variant v in {fixed, learned, name}): outputs/cage_<d>_<v>/seed_<s>/metrics.json
  fixed   = description text, fixed fusion      learned = description text, learned fusion
  name    = category NAME text, fixed fusion (ablation; outputs/cage_name_<d>_fixed, session-1 outputs are renamed to this)

Reports (all micro-F1 in %, sample std ddof=1):
 1. stability: per domain/variant mean+-std, 95% t-interval, min-max, seeds above best baseline, margin,
    one-sided one-sample t-test vs the baseline's single run (baseline variance ignored, so optimistic).
 2. configuration selection, decided on DEV only: learned is chosen for a domain iff its mean dev F1 exceeds
    fixed by more than 1 SD of fixed's dev F1; otherwise fixed (simpler, no extra parameters).
 3. fixed vs learned, paired by seed (test): mean difference, seeds where learned wins, paired two-sided t-test.
 4. name vs description (fixed fusion), paired by seed (test): same statistics.
 5. design ablations gate_hard / gate_none / query_id (outputs/cage_abl_<abl>_<d>_fixed), paired vs CAGE fixed.
 --latex DIR additionally writes DIR/tab_stability.tex, tab_selection.tex, tab_name_vs_desc.tex

Usage: python3 scripts/cage_stats.py [--out_dir outputs] [--json paper/error_analysis/cage_5seed_stats.json] [--latex paper/tables]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

BEST_BASELINE = {  # best single-run baseline per domain: (name, micro-F1 %)
    "restaurant": ("NL Instruction-En", 77.74), "hotel": ("NL Instruction-Vi", 74.83),
    "phone": ("Code Instruction-Vi", 84.35), "education": ("Ensemble BERTs", 83.81),
    "beauty": ("Ensemble BERTs", 92.45)}
KEY = "acsa_f1_micro"
DISPLAY = {"restaurant": "Restaurant", "hotel": "Hotel", "phone": "Phone", "education": "Education", "beauty": "Beauty"}


def load(out_dir, d, v):
    """{seed: {'test': F1, 'dev': F1}} in %, or None if the run is missing."""
    if v == "name":
        base = Path(out_dir) / f"cage_name_{d}_fixed"
    elif v.startswith("abl:"):
        base = Path(out_dir) / f"cage_abl_{v[4:]}_{d}_fixed"
    else:
        base = Path(out_dir) / f"cage_{d}_{v}"
    seeds = sorted(base.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    res = {}
    for s in seeds:
        f = s / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            sc = lambda x: x * 100 if x <= 1.0 else x
            res[int(s.name.split("_")[1])] = {"test": sc(m["test"][KEY]), "dev": sc(m["dev"][KEY])}
    return res or None


def ms(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else float("nan"))


def stability(x, ref):
    x = np.asarray(x, float); n = len(x); m, s = ms(x)
    half = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    t, p = stats.ttest_1samp(x, ref, alternative="greater")
    return {"n": n, "mean": m, "std": s, "ci95": [m - half, m + half], "min": x.min(), "max": x.max(),
            "seeds_above": int((x > ref).sum()), "margin": m - ref, "t": float(t), "p_one_sided": float(p), "values": x.tolist()}


def paired(a, b):
    """a, b: {seed: F1}; statistics of a - b over shared seeds."""
    seeds = sorted(set(a) & set(b)); d = np.array([a[s] - b[s] for s in seeds])
    t, p = stats.ttest_1samp(d, 0.0) if len(d) > 1 else (float("nan"), float("nan"))
    return {"n": len(d), "mean_diff": float(d.mean()), "a_wins": int((d > 0).sum()), "t": float(t), "p_two_sided": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--json", default="paper/error_analysis/cage_5seed_stats.json")
    ap.add_argument("--latex", default=None)
    a = ap.parse_args()
    R, runs = {}, {}
    for d in BEST_BASELINE:
        for v in ("fixed", "learned", "name"):
            runs[d, v] = load(a.out_dir, d, v)

    print("1. STABILITY vs best single-run baseline (test micro-F1)")
    print(f"{'domain':11s}{'variant':9s}{'mean±std':>13s}{'95% CI':>18s}{'min-max':>14s}{'>base':>6s}{'margin':>8s}{'p':>8s}  baseline")
    for d, (bn, b) in BEST_BASELINE.items():
        for v in ("fixed", "learned"):
            r = runs[d, v]
            if not r: print(f"{d:11s}{v:9s}  (missing)"); continue
            st = stability([x["test"] for x in r.values()], b); st["baseline"] = {"name": bn, "f1": b}
            R[f"stability/{d}/{v}"] = st
            print(f"{d:11s}{v:9s}{st['mean']:7.2f}±{st['std']:.2f}  [{st['ci95'][0]:.2f},{st['ci95'][1]:.2f}]  "
                  f"{st['min']:.2f}-{st['max']:.2f}  {st['seeds_above']}/{st['n']}{st['margin']:+8.2f}{st['p_one_sided']:8.4f}  {bn} {b}")

    print("\n2. CONFIG SELECTION on DEV (learned iff dev mean > fixed dev mean + 1 SD of fixed dev)")
    for d in BEST_BASELINE:
        f, l = runs[d, "fixed"], runs[d, "learned"]
        if not (f and l): continue
        fm, fs = ms([x["dev"] for x in f.values()]); lm, _ = ms([x["dev"] for x in l.values()])
        choice = "learned" if lm > fm + fs else "fixed"
        R[f"selection/{d}"] = {"dev_fixed": fm, "dev_fixed_std": fs, "dev_learned": lm, "chosen": choice}
        pr = paired({s: x["test"] for s, x in l.items()}, {s: x["test"] for s, x in f.items()})
        R[f"fixed_vs_learned/{d}"] = pr
        print(f"{d:11s} dev fixed {fm:.2f}±{fs:.2f} | learned {lm:.2f} -> {choice:7s} | test learned-fixed {pr['mean_diff']:+.2f}, "
              f"learned wins {pr['a_wins']}/{pr['n']}, paired p={pr['p_two_sided']:.3f}")

    print("\n3. NAME vs DESCRIPTION (fixed fusion, paired by seed, test)")
    for d in BEST_BASELINE:
        f, n = runs[d, "fixed"], runs[d, "name"]
        if not (f and n): print(f"{d:11s}  (need both fixed and name runs)"); continue
        (fm, fs), (nm, ns) = ms([x["test"] for x in f.values()]), ms([x["test"] for x in n.values()])
        pr = paired({s: x["test"] for s, x in f.items()}, {s: x["test"] for s, x in n.items()})
        R[f"name_vs_desc/{d}"] = {"desc_mean": fm, "desc_std": fs, "name_mean": nm, "name_std": ns, **pr}
        print(f"{d:11s} desc {fm:.2f}±{fs:.2f} | name {nm:.2f}±{ns:.2f} | desc-name {pr['mean_diff']:+.2f}, "
              f"desc wins {pr['a_wins']}/{pr['n']}, paired p={pr['p_two_sided']:.3f}")

    print("\n4. DESIGN ABLATIONS vs CAGE (fixed fusion, description text; paired by seed, test)")
    for abl in ("gate_hard", "gate_none", "query_id"):
        for d in BEST_BASELINE:
            f, x = runs[d, "fixed"], load(a.out_dir, d, f"abl:{abl}")
            if not (f and x): continue
            (fm, fs), (xm, xs) = ms([v["test"] for v in f.values()]), ms([v["test"] for v in x.values()])
            pr = paired({sd: v["test"] for sd, v in f.items()}, {sd: v["test"] for sd, v in x.items()})
            R[f"ablation/{abl}/{d}"] = {"cage_mean": fm, "cage_std": fs, "abl_mean": xm, "abl_std": xs, **pr}
            print(f"{abl:10s}{d:11s} CAGE {fm:.2f}±{fs:.2f} | ablated {xm:.2f}±{xs:.2f} | CAGE-ablated {pr['mean_diff']:+.2f}, "
                  f"CAGE wins {pr['a_wins']}/{pr['n']}, paired p={pr['p_two_sided']:.3f}")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))
    print("\nwrote", a.json)

    if a.latex:
        L = Path(a.latex); L.mkdir(parents=True, exist_ok=True)
        rows = []
        for d, (bn, b) in BEST_BASELINE.items():
            for v in ("fixed", "learned"):
                s = R.get(f"stability/{d}/{v}")
                if s:
                    rows.append(f"{DISPLAY[d]} & {v.capitalize()} & ${s['mean']:.2f}\\pm{s['std']:.2f}$ & [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}] & "
                                f"{s['min']:.2f}--{s['max']:.2f} & {s['seeds_above']}/{s['n']} & {b:.2f} & ${s['margin']:+.2f}$ & {s['p_one_sided']:.3f} \\\\")
        (L / "tab_stability.tex").write_text("\n".join(rows) + "\n")
        rows = []
        for d in BEST_BASELINE:
            s = R.get(f"selection/{d}"); p = R.get(f"fixed_vs_learned/{d}")
            if s and p:
                rows.append(f"{DISPLAY[d]} & {s['dev_fixed']:.2f} & {s['dev_learned']:.2f} & {s['chosen'].capitalize()} & ${p['mean_diff']:+.2f}$ & {p['a_wins']}/{p['n']} & {p['p_two_sided']:.3f} \\\\")
        (L / "tab_selection.tex").write_text("\n".join(rows) + "\n")
        rows = []
        for d in BEST_BASELINE:
            s = R.get(f"name_vs_desc/{d}")
            if s:
                rows.append(f"{DISPLAY[d]} & ${s['name_mean']:.2f}\\pm{s['name_std']:.2f}$ & ${s['desc_mean']:.2f}\\pm{s['desc_std']:.2f}$ & ${s['mean_diff']:+.2f}$ & {s['a_wins']}/{s['n']} & {s['p_two_sided']:.3f} \\\\")
        (L / "tab_name_vs_desc.tex").write_text("\n".join(rows) + "\n")
        print("wrote LaTeX rows to", L)


if __name__ == "__main__":
    main()
