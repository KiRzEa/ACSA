"""Recover the NL-format instruction-tuned baseline runs broken by the ViT5-base '#'-token bug
(baselines/t5_instruction_tuning.py, canonicalize_category docstring has the full diagnosis).

No retraining needed: the saved test_predictions.jsonl already has the model's category guesses,
just missing '#' (e.g. 'FOODQUALITY' instead of 'FOOD#QUALITY'). This rewrites every affected
prediction file in place with canonicalized category names, and its multi_seed_summary.json with
the recomputed metric.

Usage: python3 scripts/fix_nl_instruction_predictions.py [--out_dir outputs] [--dry_run]
Only touches nl_vi/nl_en runs (code_vi/code_en use CodeT5-base, which is unaffected).
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_mtl_acsa_v2 import parse_dataset  # noqa: E402
sys.path.insert(0, str(ROOT / "baselines"))
from t5_instruction_tuning import canonicalize_category  # noqa: E402
from common import micro_prf  # noqa: E402

DOMAINS = {"restaurant": "Res_ABSA", "hotel": "Hotel_ABSA", "phone": "Phone_ABSA",
           "education": "Education_ABSA", "beauty": "Beauty_ABSA"}


def domain_categories(domain):
    ex = parse_dataset(ROOT / DOMAINS[domain] / "Train.txt") + parse_dataset(ROOT / DOMAINS[domain] / "Dev.txt") \
        + parse_dataset(ROOT / DOMAINS[domain] / "Test.txt")
    return sorted({c for e in ex for c, _ in e.labels})


def fix_file(path, categories):
    lines = path.read_text(encoding="utf-8").splitlines()
    fixed, changed = [], 0
    for line in lines:
        r = json.loads(line)
        new_pred = [{"category": canonicalize_category(p["category"], categories), "sentiment": p["sentiment"]}
                    for p in r["prediction"]]
        if new_pred != r["prediction"]:
            changed += 1
        r["prediction"] = new_pred
        fixed.append(r)
    gold = [[(p["category"], p["sentiment"]) for p in r["gold"]] for r in fixed]
    pred = [[(p["category"], p["sentiment"]) for p in r["prediction"]] for r in fixed]
    metrics = micro_prf(gold, pred)
    return fixed, changed, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    total_fixed = 0
    for d in DOMAINS:
        cats = domain_categories(d)
        for lang in ("vi", "en"):
            base = out_dir / f"t5_{d}_nl_{lang}"
            if not base.exists():
                continue
            run_dirs = sorted(base.glob("seed_*")) or ([base] if (base / "test_predictions.jsonl").exists() else [])
            per_seed = []
            for rd in run_dirs:
                pf = rd / "test_predictions.jsonl"
                if not pf.exists():
                    continue
                fixed, changed, metrics = fix_file(pf, cats)
                seed = int(rd.name.split("_")[1]) if rd.name.startswith("seed_") else 42
                print(f"{d:11s} nl_{lang} seed {seed:5d}: {changed:4d}/{len(fixed)} predictions had '#' restored, "
                      f"F1 {metrics['f1']:.2f} (P {metrics['precision']:.2f} / R {metrics['recall']:.2f})")
                per_seed.append({"seed": seed, **metrics})
                if not a.dry_run and changed:
                    pf.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fixed) + "\n", encoding="utf-8")
                    summary_path = rd / "multi_seed_summary.json"
                    if summary_path.exists():
                        s = json.loads(summary_path.read_text())
                        s["best"] = metrics
                        s["per_seed"] = [{"seed": seed, **metrics}]
                        summary_path.write_text(json.dumps(s, indent=2, ensure_ascii=False))
                    total_fixed += changed
    if a.dry_run:
        print("\n(dry run -- no files written; drop --dry_run to apply)")
    else:
        print(f"\nrewrote {total_fixed} predictions across affected files")


if __name__ == "__main__":
    main()
