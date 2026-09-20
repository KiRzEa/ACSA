"""Where do the errors come from: deciding WHICH categories apply, or the polarity?

For one or more test_predictions.jsonl files (project schema) print
  joint micro-F1 (category+polarity), category-only micro-F1 (polarity ignored),
  polarity accuracy given a correctly detected category, and gold/predicted counts per category.
Usage: python3 scripts/subtask_breakdown.py FILE [FILE ...] [--per_category]
"""
import argparse, json
from collections import Counter


def analyse(path):
    tp = fp = fn = jt = jp = jg = ok = 0
    gc, pc, hc = Counter(), Counter(), Counter()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g = {x["category"]: x["sentiment"] for x in r["gold"]}
        p = {x["category"]: x["sentiment"] for x in r["prediction"]}
        jg += len(g); jp += len(p); gc.update(list(g)); pc.update(list(p))
        for c, s in p.items():
            if c in g:
                tp += 1; hc[c] += 1; ok += g[c] == s; jt += g[c] == s
            else:
                fp += 1
        fn += sum(c not in p for c in g)
    f = lambda t, a, b: 200 * t / (a + b) if a + b else 0.0   # F1 = 2TP/(2TP+FP+FN) = 2t/(pred+gold)
    return {"joint_F1": f(jt, jp, jg), "category_only_F1": f(tp, tp + fp, tp + fn),
            "polarity_acc_given_category": 100 * ok / tp if tp else 0.0,
            "gold_pairs": jg, "pred_pairs": jp}, gc, pc, hc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--per_category", action="store_true")
    a = ap.parse_args()
    for f in a.files:
        s, gc, pc, hc = analyse(f)
        print(f"{f}\n  joint F1 {s['joint_F1']:.2f} | category-only F1 {s['category_only_F1']:.2f} | "
              f"polarity acc given category {s['polarity_acc_given_category']:.1f}% | gold {s['gold_pairs']} pred {s['pred_pairs']}")
        if a.per_category:
            for c in sorted(set(gc) | set(pc), key=lambda c: -gc[c]):
                print(f"    {c:32s} gold {gc[c]:4d}  pred {pc[c]:4d}  category-hit {hc[c]:4d}")
