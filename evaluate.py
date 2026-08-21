import json
from collections import defaultdict


def evaluate_jsonl_per_category(jsonl_path):
    gold_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            gold = {g["category"]: g["sentiment"] for g in record["gold"]}
            pred = {p["category"]: p["sentiment"] for p in record["prediction"]}

            for cat in gold:
                gold_counts[cat] += 1
            for cat, sent in pred.items():
                pred_counts[cat] += 1
                if gold.get(cat) == sent:
                    correct_counts[cat] += 1

    categories = sorted(set(gold_counts) | set(pred_counts))
    for cat in categories:
        c, p, g = correct_counts[cat], pred_counts[cat], gold_counts[cat]
        prec = 100 * c / p if p else 0.0
        rec = 100 * c / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(cat)
        print("%0.2f\t%0.2f\t%0.2f" % (prec, rec, f1))

    tot_c, tot_p, tot_g = sum(correct_counts.values()), sum(pred_counts.values()), sum(gold_counts.values())
    P = 100 * tot_c / tot_p if tot_p else 0.0
    R = 100 * tot_c / tot_g if tot_g else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    print("-" * 60)
    print("Mean Precision score: ", round(P, 2))
    print("Mean Recall score: ", round(R, 2))
    print("Mean F1 score: ", round(F, 2))
    return P, R, F


if __name__ == "__main__":
    evaluate_jsonl_per_category("phobert_mtl_acsa/test_predictions.jsonl")