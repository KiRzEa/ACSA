#!/usr/bin/env python3
"""Six-agent CoT prompting baseline for category-level ACSA, ADAPTED from Ventirozos, Appleby and Shardlow (2025),
"Are You Sure You're Positive? Consolidating Chain-of-Thought Agents with Uncertainty Quantification for ACSA"
(arXiv 2508.17258, Appendix E). Not a reproduction: different LLMs (see --model), Vietnamese data, category
descriptions added to the category list, and no token-logprob aggregation (reasoning models may not return logprobs).

Each review is sent to six agents; agent k follows one of the six orders of the elements Aspect (A), Opinion (O)
and Category (C) inside one enumerated prompt, then outputs a Python list of (category, polarity) tuples.
Predictions are aggregated offline from raw.jsonl:
  most_common  the list produced by most agents (ties: first agent in ORDERS)      [paper: "most common list"]
  union        all pairs of all agents, per-category majority polarity              [paper: "joined CoT agent"]
  majority     pair kept when at least 3 of the 6 agents produced it                [our addition]
  agent_<order> each single agent, for the element-order analysis
Outputs in outputs_llm/cot6_<domain>_<model>[_smoke]/: raw.jsonl, test_predictions_<agg>.jsonl, summary.json

Usage: python3 llm_baseline/run_llm_cot6.py --provider azure --model Llama-3.3-70B-Instruct --domain education [--limit 20]
"""
import argparse, ast, contextlib, difflib, io, itertools, json, os, random, re, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "llm_preprocessing")); sys.path.insert(0, str(ROOT / "llm_baseline"))
import llm_client as L                          # noqa: E402  (loads .env)
import mapper                                   # noqa: E402
from evaluate import evaluate_jsonl_per_category  # noqa: E402
from run_llm_acsa import DATA, DOMAIN_TEXT, POL, load  # noqa: E402

ORDERS = ["".join(p) for p in itertools.permutations("ACO")]        # 6 orders of Aspect, Category, Opinion
SYSTEM = ("You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. I want you to force "
          "yourself to pick words that you are being asked and only them, without explanations or reasoning. If you are unsure, "
          "put the most probable. Now follow the following steps:")
NAME = {"A": "Aspects", "O": "Opinions", "C": "Categories"}
SING = {"A": "aspects", "O": "opinions", "C": "categories"}


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


def step_text(el, first, prev, dom, text, cats):
    if el == "A":
        return (f'Given the following text, list all word sequences that denote an aspect term of the {dom} domain:\n"{text}"' if first
                else f"List all word sequences that denote an aspect term from the {SING[prev]} detected.")
    if el == "O":
        return (f'Given the following text, list all word sequences that denote an opinion of the {dom} domain:\n"{text}"' if first
                else f"List all word sequences that denote or link to an opinion from the {SING[prev]} detected.")
    lead = f'Given the following text, list the categories it evaluates:\n"{text}"' if first else f"List the categories from the {SING[prev]} detected."
    return f"{lead}\nThe list of possible categories (CODE: what it covers) is:\n{cats}"


def build_prompt(order, dom, text, cats):
    steps = [f"{i + 1}. " + step_text(el, i == 0, order[i - 1] if i else None, dom, text, cats) for i, el in enumerate(order)]
    blanks = [f"{i + 1}. {NAME[el]}:" for i, el in enumerate(order)]
    last = ("Lastly, please provide one Python-type list of tuples such as\n\"[('example_category_1', 'positive'), ('example_category_2', 'negative'), ...]\"\n"
            "that you identified, using only the category CODEs above. The sentiment is either 'positive', 'neutral' or 'negative', "
            "based on the extracted opinions. If no category applies, answer [].")
    return "\n\n".join(steps) + "\n" + "-" * 20 + "\n" + "\n".join(blanks) + "\n\n" + last


def parse_pairs(content, codes):
    """Last Python-style list of (category, polarity) tuples in the answer; categories snapped to valid codes with difflib."""
    if not content:
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    cand = re.findall(r"\[\s*(?:\(.*?\)\s*,?\s*)*\]", content, flags=re.S)
    pairs = []
    src = cand[-1] if cand else content
    for c, s in re.findall(r"\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([A-Za-z]+)['\"]\s*\)", src):
        s = s.lower()
        if s not in POL:
            continue
        if c not in codes:
            m = difflib.get_close_matches(c, codes, n=1, cutoff=0.6)
            if not m:
                continue
            c = m[0]
        pairs.append((c, s))
    return pairs


def as_dict(pairs):
    d = {}
    for c, s in pairs or []:
        d.setdefault(c, s)
    return d


def aggregate(lists):
    """lists: 6 pair-lists (None when unparsable) in ORDERS order -> dict aggregator -> {category: polarity}."""
    ok = [l for l in lists if l is not None]
    out = {}
    if not ok:
        return {"most_common": {}, "union": {}, "majority": {}}
    keys = [frozenset(as_dict(l).items()) for l in ok]
    cnt = Counter(keys); best = max(cnt.values())
    out["most_common"] = dict(next(k for k in keys if cnt[k] == best))
    pol = {}
    for l in ok:
        for c, s in as_dict(l).items():
            pol.setdefault(c, Counter())[s] += 1
    out["union"] = {c: cn.most_common(1)[0][0] for c, cn in pol.items()}
    pair_cnt = Counter(itertools.chain.from_iterable(as_dict(l).items() for l in ok))
    out["majority"] = {}
    for (c, s), n in pair_cnt.most_common():
        if n >= 3:
            out["majority"].setdefault(c, s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="azure", choices=["openai", "azure", "fpt"])
    ap.add_argument("--model", required=True, help="deployment name (Azure) / model name")
    ap.add_argument("--domain", required=True, choices=list(DATA))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rpm", type=int, default=0, help="cap on requests per minute (0 = unlimited); Azure Llama-3.3 free quota is 20 RPM / 20k TPM")
    ap.add_argument("--reasoning_effort", default="low", choices=["none", "minimal", "low", "medium", "high"])
    ap.add_argument("--price_in", type=float, default=None)
    ap.add_argument("--price_out", type=float, default=None)
    a = ap.parse_args()
    out = ROOT / "outputs_llm" / f"cot6_{a.domain}_{a.model}{'_smoke' if a.limit else ''}"; out.mkdir(parents=True, exist_ok=True)

    train, test = load(a.domain, "Train"), load(a.domain, "Test")
    codes = sorted({c for e in train for c, _ in e.labels}); desc = mapper.get_category_descriptions(codes, a.domain)
    cats = "\n".join(f"- {c}: {desc[c]}" for c in codes); dom = DOMAIN_TEXT[a.domain]
    if a.limit:
        test = random.Random(0).sample(test, min(a.limit, len(test)))

    from openai import OpenAI
    env = {"openai": (None, "OPENAI_API_KEY"), "azure": ("AZURE_FOUNDRY_ENDPOINT", "AZURE_API_KEY"), "fpt": ("FPT_URL", "FPT_API_KEY")}[a.provider]
    client = OpenAI(base_url=os.environ[env[0]] if env[0] else None, api_key=os.environ[env[1]], timeout=180)
    lock = threading.Lock(); limiter = RateLimiter(a.rpm)
    send = {"store": a.provider in ("openai", "azure"), "effort": a.reasoning_effort != "none" and any(k in a.model.lower() for k in ("gpt-5", "gpt-oss", "o3", "o4")),
            "rf": True}

    raw_path = out / "raw.jsonl"; done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("content") is not None:
                done[(r["id"], r["order"])] = r

    def call(e, order):
        last = None
        for attempt in range(8):
            try:
                limiter.wait()
                kw = dict(model=a.model, messages=[{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": build_prompt(order, dom, e.text, cats)}])
                if send["store"]: kw["store"] = False
                if send["effort"]: kw["reasoning_effort"] = a.reasoning_effort
                if not any(k in a.model.lower() for k in ("gpt-5", "o3", "o4")): kw["temperature"] = 0
                u = None
                resp = client.chat.completions.create(**kw); u = resp.usage
                return {"id": e.sample_id, "order": order, "content": resp.choices[0].message.content,
                        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
            except Exception as ex:
                last = ex; msg = str(ex).lower()
                if "reasoning_effort" in msg: send["effort"] = False
                elif "store" in msg and ("unsupported" in msg or "unknown" in msg or "unrecognized" in msg): send["store"] = False
                time.sleep(min(2 ** attempt, 60) if "429" in msg or "rate" in msg else min(2 ** attempt, 30))
        return {"id": e.sample_id, "order": order, "content": None, "error": str(last)[:200]}

    todo = [(e, o) for e in test for o in ORDERS if (e.sample_id, o) not in done]
    print(f"{a.domain}: {len(test)} reviews x 6 agents = {len(todo)} calls to make, model={a.model} ({a.provider})", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, raw_path.open("a", encoding="utf-8") as f:
        futs = [ex.submit(call, e, o) for e, o in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if r.get("content") is not None: done[(r["id"], r["order"])] = r
            if i % 300 == 0: print(f"  {i}/{len(todo)}", flush=True)

    preds = {k: [] for k in ["most_common", "union", "majority"] + [f"agent_{o}" for o in ORDERS]}
    unparsable = 0
    for e in test:
        lists = []
        for o in ORDERS:
            r = done.get((e.sample_id, o)); l = parse_pairs(r["content"], codes) if r else None
            unparsable += l is None; lists.append(l)
        agg = aggregate(lists)
        for k in ("most_common", "union", "majority"): preds[k].append(agg[k])
        for o, l in zip(ORDERS, lists): preds[f"agent_{o}"].append(as_dict(l))
    scores = {}
    for k, plist in preds.items():
        path = out / f"test_predictions_{k}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e, p in zip(test, plist):
                f.write(json.dumps({"id": e.sample_id, "text": e.text, "gold": [{"category": c, "sentiment": s} for c, s in e.labels],
                                    "prediction": [{"category": c, "sentiment": s} for c, s in p.items()]}, ensure_ascii=False) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            P, R, F = evaluate_jsonl_per_category(path)
        scores[k] = {"P": round(P, 2), "R": round(R, 2), "F1": round(F, 2)}
    rs = list(done.values()); tp = sum(r["prompt_tokens"] for r in rs); tc = sum(r["completion_tokens"] for r in rs)
    summary = {"domain": a.domain, "model": a.model, "provider": a.provider, "n_reviews": len(test), "calls": len(rs), "unparsable_calls": unparsable,
               "prompt_tokens": tp, "completion_tokens": tc, "avg_prompt_per_call": tp / max(len(rs), 1), "avg_completion_per_call": tc / max(len(rs), 1),
               "cost_usd": ((tp * a.price_in + tc * a.price_out) / 1e6 if a.price_in is not None and a.price_out is not None else None), "scores": scores}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
