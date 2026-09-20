#!/usr/bin/env python3
"""Zero-shot LLM prompting baseline for category-level ACSA (own design, in the spirit of
Ventirozos et al. 2025: zero-shot ACSA with structured reasoning). Not a reproduction.

For each test review the model gets the domain's category list with the mapper descriptions and
returns {"predictions": [{"category", "evidence", "sentiment"}]}. Output is written in the project's
test_predictions.jsonl schema and scored with evaluate.evaluate_jsonl_per_category (micro P/R/F1).

Usage:
  python3 llm_baseline/run_llm_acsa.py --domain hotel --limit 40          # smoke test (random, seeded)
  python3 llm_baseline/run_llm_acsa.py --domain hotel                     # full test set
Resumable: finished ids are cached in <out>/raw.jsonl.  Cost is logged to <out>/cost.json.
"""
import argparse, contextlib, io, json, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "llm_preprocessing"))
import llm_client as L                       # noqa: E402
import mapper                                # noqa: E402
from evaluate import evaluate_jsonl_per_category  # noqa: E402

DATA = {"restaurant": "Res_ABSA", "hotel": "Hotel_ABSA", "phone": "Phone_ABSA",
        "education": "Education_ABSA", "beauty": "Beauty_ABSA"}
DOMAIN_TEXT = {"restaurant": "restaurant", "hotel": "hotel", "phone": "mobile phone",
               "education": "teaching and course", "beauty": "beauty product"}
POL = {"positive", "neutral", "negative"}

SYSTEM = """You are an expert annotator for aspect-category sentiment analysis (ACSA) of Vietnamese {dom} reviews.
Given one review, decide which of the predefined aspect categories it evaluates and the sentiment toward each.

Categories (CODE: what it covers):
{cats}

Rules:
- Use only the CODEs above, copied exactly. A category counts when the review expresses an opinion about it, explicitly or implicitly.
- Give at most one sentiment per category: positive, neutral or negative. If opinions on one category conflict, choose the dominant one.
- Do not invent categories the review does not evaluate. If none applies, return an empty list.
- For each prediction, "evidence" is a short quote (max 12 words) from the review that supports it; write the quote first, then decide the sentiment.
Answer with JSON only: {{"predictions": [{{"category": "<CODE>", "evidence": "<quote>", "sentiment": "positive|neutral|negative"}}]}}"""


def load(domain, split):
    import train_mtl_acsa_v2 as T
    return T.parse_dataset(ROOT / DATA[domain] / f"{split}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DATA))
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--reasoning_effort", default="low", choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0, help="random subset of the test set (seed 0); 0 = all")
    ap.add_argument("--fewshot", type=int, default=0,
                    help="K training examples per category shown in the prompt (seeded, from Train only); 0 = zero-shot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or ROOT / "outputs_llm" / f"baseline_{a.domain}_{a.model}{f'_fewshot{a.fewshot}' if a.fewshot else ''}{'_smoke' if a.limit else ''}")
    out.mkdir(parents=True, exist_ok=True)

    train, test = load(a.domain, "Train"), load(a.domain, "Test")
    cats = sorted({c for e in train for c, _ in e.labels})
    desc = mapper.get_category_descriptions(cats, a.domain)
    system = SYSTEM.format(dom=DOMAIN_TEXT[a.domain], cats="\n".join(f"- {c}: {desc[c]}" for c in cats))
    if a.fewshot:  # K short training reviews per category (never Dev/Test), shown with their gold labels
        rng, shots, seen = random.Random(0), [], set()
        pool = [e for e in train if len(e.text.split()) <= 40]
        for c in cats:
            cand = [e for e in pool if any(l[0] == c for l in e.labels) and e.sample_id not in seen]
            for e in rng.sample(cand, min(a.fewshot, len(cand))):
                seen.add(e.sample_id); shots.append(e)
        system += "\n\nLabeled examples from the training set (review => CODE=sentiment; the same category conventions apply to the review you will receive):\n" + "\n".join(
            f"Review: {e.text}\n=> " + "; ".join(f"{c}={s}" for c, s in e.labels) for e in shots)
        print(f"few-shot: {len(shots)} examples from Train")
    if a.limit:
        test = random.Random(0).sample(test, min(a.limit, len(test)))

    raw_path = out / "raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[r["id"]] = r
    tracker = L.get_tracker(out / "cost.json")
    client, lock = L.get_openai_client(), threading.Lock()

    def call(e):
        last = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=a.model, reasoning_effort=a.reasoning_effort,
                    store=False,  # do not store the completion in the OpenAI dashboard logs
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": e.text}])
                u = resp.usage
                tracker.record(a.model, u.prompt_tokens, u.completion_tokens)
                rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                return {"id": e.sample_id, "content": resp.choices[0].message.content,
                        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "reasoning_tokens": rt}
            except Exception as ex:  # rate limit / transient
                last = ex; time.sleep(min(2 ** attempt, 30))
        return {"id": e.sample_id, "content": None, "error": str(last)[:200]}

    todo = [e for e in test if e.sample_id not in done or done[e.sample_id].get("content") is None]
    print(f"{a.domain}: {len(test)} reviews, {len(cats)} categories, {len(todo)} to query, model={a.model}, effort={a.reasoning_effort}")
    with ThreadPoolExecutor(a.workers) as ex, raw_path.open("a", encoding="utf-8") as f:
        futs = [ex.submit(call, e) for e in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); done[r["id"]] = r
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0: print(f"  {i}/{len(todo)}  {tracker.summary()}", flush=True)

    bad = invalid = 0
    pred_path = out / "test_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for e in test:
            r = done.get(e.sample_id, {}); pred = {}
            try:
                for p in json.loads(r["content"]).get("predictions", []):
                    c, s = p.get("category"), str(p.get("sentiment", "")).lower()
                    if c in desc and s in POL:
                        pred.setdefault(c, s)
                    else:
                        invalid += 1
            except Exception:
                bad += 1
            f.write(json.dumps({"id": e.sample_id, "text": e.text,
                                "gold": [{"category": c, "sentiment": s} for c, s in e.labels],
                                "prediction": [{"category": c, "sentiment": s} for c, s in pred.items()]},
                               ensure_ascii=False) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        P, R, F = evaluate_jsonl_per_category(pred_path)
    rs = [done[e.sample_id] for e in test if done.get(e.sample_id, {}).get("content")]
    n = max(len(rs), 1)
    stats = {"domain": a.domain, "model": a.model, "effort": a.reasoning_effort, "n": len(test), "P": P, "R": R, "F1": F,
             "unparsable": bad, "invalid_pairs": invalid,
             "avg_prompt_tokens": sum(r["prompt_tokens"] for r in rs) / n,
             "avg_completion_tokens": sum(r["completion_tokens"] for r in rs) / n,
             "avg_reasoning_tokens": sum(r["reasoning_tokens"] for r in rs) / n, "cost_usd_logged": tracker.totals["cost_usd"]}
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
