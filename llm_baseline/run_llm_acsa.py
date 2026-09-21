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
import argparse, contextlib, io, json, random, re, sys, threading, time
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


class RateLimiter:
    """Global requests-per-minute cap shared by all worker threads (0 = unlimited)."""
    def __init__(self, rpm):
        self.gap = 60.0 / rpm if rpm else 0.0; self.next = 0.0; self.lock = threading.Lock()
    def wait(self):
        if not self.gap:
            return
        with self.lock:
            now = time.monotonic(); t = max(now, self.next); self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def load(domain, split):
    import train_mtl_acsa_v2 as T
    return T.parse_dataset(ROOT / DATA[domain] / f"{split}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DATA))
    ap.add_argument("--provider", default="openai", choices=["openai", "azure", "fpt"],
                    help="openai: OPENAI_API_KEY; azure: AZURE_FOUNDRY_ENDPOINT + AZURE_API_KEY; fpt: FPT_URL + FPT_API_KEY")
    ap.add_argument("--model", default="gpt-5-nano", help="model / deployment name as the endpoint lists it")
    ap.add_argument("--price_in", type=float, default=None, help="USD per 1M input tokens (non-openai providers)")
    ap.add_argument("--price_out", type=float, default=None, help="USD per 1M output tokens (non-openai providers)")
    ap.add_argument("--reasoning_effort", default="low", choices=["none", "minimal", "low", "medium", "high"],
                    help="sent only when the model supports it (gpt-5*, gpt-oss*); none = never send")
    ap.add_argument("--limit", type=int, default=0, help="random subset of the test set (seed 0); 0 = all")
    ap.add_argument("--fewshot", type=int, default=0,
                    help="K training examples per category shown in the prompt (seeded, from Train only); 0 = zero-shot")
    ap.add_argument("--retrieval_k", type=int, default=0,
                    help="show the K most similar labeled TRAIN reviews (char n-gram TF-IDF cosine) with their gold labels; 0 = off")
    ap.add_argument("--stats_prompt", action="store_true",
                    help="add label statistics of the TRAIN set to the system prompt (labels per review, share of reviews per category, polarity shares)")
    ap.add_argument("--stop_after_tokens", type=int, default=0,
                    help="budget guard: stop sending new requests once prompt+completion tokens of this run exceed the value (0 = no guard)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=0, help="cap on requests per minute (0 = unlimited)")
    ap.add_argument("--json_mode", default="auto", choices=["auto", "on", "off"],
                    help="send response_format=json_object; auto = off for gpt-oss (its JSON mode returned empty or malformed answers in our test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or ROOT / "outputs_llm" / f"baseline_{a.domain}_{a.model}{f'_fewshot{a.fewshot}' if a.fewshot else ''}{f'_rfs{a.retrieval_k}' if a.retrieval_k else ''}{'_stats' if a.stats_prompt else ''}{'_smoke' if a.limit else ''}")
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
    if a.stats_prompt:  # label statistics of the training set only
        from collections import Counter
        n = len(train); per = Counter(c for e in train for c, _ in e.labels); one = sum(len(e.labels) == 1 for e in train) / n
        pol = Counter(sv for e in train for _, sv in e.labels); tot = sum(pol.values())
        system += ("\n\nAnnotation statistics of this dataset (training set, use them as priors on how liberally to assign categories): "
                   f"a review carries {sum(len(e.labels) for e in train) / n:.2f} categories on average and {100 * one:.0f}% of reviews carry exactly one. "
                   "Share of reviews carrying each category: " + ", ".join(f"{c} {100 * per[c] / n:.0f}%" for c in cats)
                   + ". Polarity shares among labels: " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in pol.most_common()) + ".")
    if a.limit:
        test = random.Random(0).sample(test, min(a.limit, len(test)))
    neighbours = {}
    if a.retrieval_k:  # nearest labeled training reviews per test review (Train only, deterministic)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=True)
        xt = vec.fit_transform([e.text for e in train]); xq = vec.transform([e.text for e in test])
        sims = (xq @ xt.T).tocsr()
        for i, e in enumerate(test):
            row = sims.getrow(i); idx = row.indices[row.data.argsort()[::-1][:a.retrieval_k]]
            neighbours[e.sample_id] = [train[j] for j in idx]
        print(f"retrieval: {a.retrieval_k} nearest train reviews per test review")

    def user_message(e):
        if not a.retrieval_k:
            return e.text
        ex = "\n".join(f"Review: {t.text}\n=> " + "; ".join(f"{c}={sv}" for c, sv in t.labels) for t in neighbours[e.sample_id])
        return ("Labeled reviews from the training set that are similar to the new one (review => CODE=sentiment; follow the same category "
                f"conventions):\n{ex}\n\nNow label this review and answer with JSON only:\n{e.text}")

    raw_path = out / "raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[r["id"]] = r
    from openai import OpenAI
    env = {"openai": (None, "OPENAI_API_KEY"), "azure": ("AZURE_FOUNDRY_ENDPOINT", "AZURE_API_KEY"), "fpt": ("FPT_URL", "FPT_API_KEY")}[a.provider]
    import os
    client = OpenAI(base_url=os.environ[env[0]] if env[0] else None, api_key=os.environ[env[1]], timeout=180)
    lock = threading.Lock(); limiter = RateLimiter(a.rpm)
    tok = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    tracker = L.get_tracker(out / "cost.json") if a.provider == "openai" else None
    sends_effort = a.reasoning_effort != "none" and any(k in a.model.lower() for k in ("gpt-5", "gpt-oss", "o3", "o4"))
    send_store = {"v": a.provider in ("openai", "azure")}   # store=False keeps completions out of the provider dashboard
    no_response_format = set() if (a.json_mode == "on" or (a.json_mode == "auto" and "gpt-oss" not in a.model.lower())) else {a.model}
    no_response_format_note = None
   # models/providers that reject response_format=json_object are remembered here

    budget_hit = {"v": False}

    def call(e):
        last = None
        if a.stop_after_tokens and tok["prompt_tokens"] + tok["completion_tokens"] > a.stop_after_tokens:
            budget_hit["v"] = True
            return {"id": e.sample_id, "content": None, "error": "budget guard"}
        for attempt in range(8):
            try:
                limiter.wait()
                kw = dict(model=a.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user_message(e)}])
                if send_store["v"]:
                    kw["store"] = False  # do not store the completion in the provider's dashboard logs
                if sends_effort:
                    kw["reasoning_effort"] = a.reasoning_effort
                if a.model not in no_response_format:
                    kw["response_format"] = {"type": "json_object"}
                if not any(k in a.model.lower() for k in ("gpt-5", "o3", "o4")):
                    kw["temperature"] = 0
                resp = client.chat.completions.create(**kw)
                u = resp.usage
                with lock:
                    tok["calls"] += 1; tok["prompt_tokens"] += u.prompt_tokens; tok["completion_tokens"] += u.completion_tokens
                if tracker is not None:
                    tracker.record(a.model, u.prompt_tokens, u.completion_tokens)
                rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                return {"id": e.sample_id, "content": resp.choices[0].message.content,
                        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "reasoning_tokens": rt}
            except Exception as ex:  # rate limit / transient / unsupported parameter
                last = ex; msg = str(ex).lower()
                if "response_format" in msg or "json_object" in msg:
                    no_response_format.add(a.model)
                elif "store" in msg and ("unsupported" in msg or "unknown" in msg or "unrecognized" in msg):
                    send_store["v"] = False  # endpoint rejects store=; stop sending it
                time.sleep(min(2 ** attempt, 30))
        return {"id": e.sample_id, "content": None, "error": str(last)[:200]}

    todo = [e for e in test if e.sample_id not in done or done[e.sample_id].get("content") is None]
    print(f"{a.domain}: {len(test)} reviews, {len(cats)} categories, {len(todo)} to query, model={a.model}, effort={a.reasoning_effort}")
    with ThreadPoolExecutor(a.workers) as ex, raw_path.open("a", encoding="utf-8") as f:
        futs = [ex.submit(call, e) for e in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); done[r["id"]] = r
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0: print(f"  {i}/{len(todo)}  " + (tracker.summary() if tracker is not None else f"{tok['calls']} calls, {tok['prompt_tokens'] + tok['completion_tokens']:,} tokens"), flush=True)

    bad = invalid = 0
    pred_path = out / "test_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for e in test:
            r = done.get(e.sample_id, {}); pred = {}
            try:
                for p in L._extract_json(re.sub(r"<think>.*?</think>", "", r["content"], flags=re.S)).get("predictions", []):
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
             "avg_reasoning_tokens": sum(r["reasoning_tokens"] for r in rs) / n, "provider": a.provider, "budget_guard_hit": budget_hit["v"], "retrieval_k": a.retrieval_k, "stats_prompt": a.stats_prompt,
             "total_prompt_tokens": sum(r["prompt_tokens"] for r in rs), "total_completion_tokens": sum(r["completion_tokens"] for r in rs),
             "cost_usd_logged": (tracker.totals["cost_usd"] if tracker is not None else
                                 ((sum(r["prompt_tokens"] for r in rs) * a.price_in + sum(r["completion_tokens"] for r in rs) * a.price_out) / 1e6
                                  if a.price_in is not None and a.price_out is not None else None))}
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
