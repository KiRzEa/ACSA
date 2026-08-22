#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset 1: raw text -> Generator/Verifier/Reflector, guideline developed on
a TRAIN sample (never Dev -- Dev stays untouched for PhoBERT's own
threshold/early-stopping), then applied frozen (no more verification) to
the full Train/Dev/Test splits.

Two modes, run separately so you can review the guideline before spending
on the full-scale pass:

    python run_dataset1.py develop --domain hotel --model gpt-4o-mini
    python run_dataset1.py apply --domain hotel --model gpt-4o-mini \\
        --guideline_file outputs_llm/hotel_dataset1/guideline_final.txt

`develop` writes a full audit trail (every round's guideline + pass rate +
failures) to outputs_llm/<domain>_dataset1/develop_log.json, plus the final
guideline to guideline_final.txt, plus an honest held-out pass rate
(measured on a separate TRAIN sample never used for refinement).

`apply` is resumable: writes one line per example to <split>.jsonl as it
goes, and skips ids already present if you re-run after an interruption.

Cost is tracked SEPARATELY per phase per domain (not one global number):
    outputs_llm/<domain>_dataset1/cost_develop.json
    outputs_llm/<domain>_dataset1/cost_apply.json
`apply` prints the combined total (develop + apply = this domain's full
Dataset 1 cost) when it finishes.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from data_io import Example, load_examples
from gvr_core import run_generator, run_verifier, run_reflector
import prompts as P
from domain_context import DOMAIN_LABELS, category_context_block
from llm_client import get_tracker, combined_cost

RAW_ROOT = {"hotel": "Hotel_ABSA", "restaurant": "Res_ABSA"}


def split_train_samples(train_examples, refine_n: int, validation_n: int, seed: int):
    rng = random.Random(seed)
    shuffled = list(train_examples)
    rng.shuffle(shuffled)
    refine_sample = shuffled[:refine_n]
    validation_sample = shuffled[refine_n:refine_n + validation_n]
    return refine_sample, validation_sample


def run_generator_verifier_batch(
    examples, guideline: str, domain: str, model: str, tracker, desc: str = "",
    provider: str = "openai", aws_region: str = "us-east-1",
):
    """Returns (pass_rate, results) where results is a list of dicts with
    id, original_text, cleaned_text, passed, issues -- for logging/reflection."""
    results = []
    n_pass = 0
    bar = tqdm(examples, desc=desc, unit="ex")
    for ex in bar:
        cleaned = run_generator(ex.text, guideline, domain, model, tracker, provider, aws_region)
        passed, issues = run_verifier(ex.text, cleaned, ex.labels, domain, model, tracker, provider, aws_region)
        if passed:
            n_pass += 1
        results.append({
            "id": ex.sample_id, "original_text": ex.text, "cleaned_text": cleaned,
            "passed": passed, "issues": issues,
        })
        bar.set_postfix(pass_rate=f"{n_pass/len(results):.2f}", cost=f"${tracker.totals['cost_usd']:.4f}")
    pass_rate = n_pass / len(examples) if examples else 0.0
    return pass_rate, results


def format_failures_for_reflector(results, max_examples: int = 15) -> str:
    fails = [r for r in results if not r["passed"]]
    blocks = []
    for r in fails[:max_examples]:
        issues_str = "; ".join(f"{i.get('category')}: {i.get('problem')} - {i.get('detail')}" for i in r["issues"])
        blocks.append(
            f"Gốc: {r['original_text']}\nĐã xử lý: {r['cleaned_text']}\nLỗi: {issues_str}"
        )
    return "\n\n".join(blocks)


def cmd_develop(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / f"{args.domain}_dataset1"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = get_tracker(output_dir / "cost_develop.json")

    train_path = Path(RAW_ROOT[args.domain]) / "Train.txt"
    train_examples = load_examples(train_path)
    print(f"Loaded {len(train_examples)} train examples from {train_path}")

    refine_sample, validation_sample = split_train_samples(
        train_examples, args.refine_sample_size, args.validation_sample_size, args.seed
    )
    print(f"Refine sample: {len(refine_sample)}, validation sample: {len(validation_sample)} (disjoint, both from Train)")

    guideline = P.INITIAL_GUIDELINE.format(
        domain_label=DOMAIN_LABELS[args.domain], category_context=category_context_block(args.domain)
    )
    log = {"domain": args.domain, "model": args.model, "rounds": []}

    for round_idx in range(1, args.max_rounds + 1):
        pass_rate, results = run_generator_verifier_batch(
            refine_sample, guideline, args.domain, args.model, tracker, desc=f"round {round_idx} refine",
            provider=args.provider, aws_region=args.aws_region,
        )
        print(f"Round {round_idx}: pass_rate={pass_rate:.3f} on refine sample ({len(refine_sample)} examples)")
        log["rounds"].append({"round": round_idx, "pass_rate": pass_rate, "guideline": guideline, "results": results})
        (output_dir / "develop_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        if pass_rate >= args.pass_threshold:
            print(f"Converged: pass_rate {pass_rate:.3f} >= threshold {args.pass_threshold}")
            break
        if round_idx == args.max_rounds:
            print(f"Reached max_rounds={args.max_rounds} without hitting threshold -- using last guideline anyway")
            break

        failures_str = format_failures_for_reflector(results)
        n_fails = sum(1 for r in results if not r["passed"])
        guideline, summary = run_reflector(
            guideline, failures_str, n_fails, len(refine_sample), args.model, tracker,
            provider=args.provider, aws_region=args.aws_region,
        )
        print(f"Reflector update: {summary[:200]}")

    # Honest held-out check: guideline was never touched by this sample.
    final_pass_rate, final_results = run_generator_verifier_batch(
        validation_sample, guideline, args.domain, args.model, tracker, desc="held-out validation",
        provider=args.provider, aws_region=args.aws_region,
    )
    print(f"Held-out validation pass_rate (never used for refinement): {final_pass_rate:.3f}")
    log["final_validation_pass_rate"] = final_pass_rate
    log["final_validation_results"] = final_results
    log["final_guideline"] = guideline
    (output_dir / "develop_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "guideline_final.txt").write_text(guideline, encoding="utf-8")
    print(f"Final guideline written to {output_dir / 'guideline_final.txt'}")
    print(f"Develop phase cost ({args.domain}): {tracker.summary()}")


def cmd_apply(args: argparse.Namespace) -> None:
    guideline = Path(args.guideline_file).read_text(encoding="utf-8")
    output_dir = Path(args.output_dir) / f"{args.domain}_dataset1"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = get_tracker(output_dir / "cost_apply.json")

    for split in ("Train", "Dev", "Test"):
        src_path = Path(RAW_ROOT[args.domain]) / f"{split}.txt"
        examples = load_examples(src_path)
        if args.limit:
            examples = examples[:args.limit]
        out_path = output_dir / f"{split}.jsonl"

        done_ids = set()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
            print(f"[{split}] Resuming: {len(done_ids)} already done")

        todo = [ex for ex in examples if ex.sample_id not in done_ids]
        write_lock = threading.Lock()
        failures_path = output_dir / f"{split}_failures.jsonl"
        n_failed = 0
        with out_path.open("a", encoding="utf-8") as f, \
             failures_path.open("a", encoding="utf-8") as ferr, \
             tqdm(total=len(todo), desc=f"apply {split}", unit="ex") as bar:

            def process(ex: Example):
                # One example failing (exhausted retries, weird API error, etc.)
                # must not kill a multi-hour run for the other ~10k examples --
                # log it to <split>_failures.jsonl and move on. Since it's never
                # written to out_path, re-running apply naturally retries it
                # (done_ids won't include it) -- no separate retry mechanism needed.
                try:
                    cleaned = run_generator(ex.text, guideline, args.domain, args.model, tracker, args.provider, args.aws_region)
                    record = {
                        "id": ex.sample_id, "text": cleaned,
                        "gold": [{"category": c, "sentiment": s} for c, s in ex.labels],
                    }
                    with write_lock:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:
                    with write_lock:
                        ferr.write(json.dumps({"id": ex.sample_id, "error": str(e)}, ensure_ascii=False) + "\n")
                        ferr.flush()
                with write_lock:
                    bar.update(1)
                    bar.set_postfix(cost=f"${tracker.totals['cost_usd']:.4f}", failed=n_failed)

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(process, ex) for ex in todo]
                for fut in as_completed(futures):
                    fut.result()  # process() itself never raises -- this just surfaces genuine bugs

        if failures_path.exists() and failures_path.stat().st_size > 0:
            n_failed = sum(1 for _ in failures_path.read_text(encoding="utf-8").splitlines() if _.strip())
            print(f"[{split}] {n_failed} examples failed, logged to {failures_path} -- re-run apply to retry them")
        print(f"[{split}] done -> {out_path}")

    print(f"Apply phase cost ({args.domain}): {tracker.summary()}")
    total = combined_cost(output_dir / "cost_develop.json", output_dir / "cost_apply.json")
    print(f"Dataset 1 TOTAL cost for {args.domain} (develop + apply): "
          f"{total['calls']} calls, {total['prompt_tokens']+total['completion_tokens']:,} tokens, ${total['cost_usd']:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    dev = sub.add_parser("develop")
    dev.add_argument("--domain", choices=["hotel", "restaurant"], required=True)
    dev.add_argument("--model", type=str, default="gpt-4o-mini")
    dev.add_argument("--provider", choices=["openai", "bedrock"], default="openai")
    dev.add_argument("--aws_region", type=str, default="us-east-1", help="Only used with --provider bedrock")
    dev.add_argument("--refine_sample_size", type=int, default=100)
    dev.add_argument("--validation_sample_size", type=int, default=200)
    dev.add_argument("--max_rounds", type=int, default=5)
    dev.add_argument("--pass_threshold", type=float, default=0.95)
    dev.add_argument("--seed", type=int, default=42)
    dev.add_argument("--output_dir", type=str, default="outputs_llm")
    dev.set_defaults(func=cmd_develop)

    app = sub.add_parser("apply")
    app.add_argument("--domain", choices=["hotel", "restaurant"], required=True)
    app.add_argument("--model", type=str, default="gpt-4o-mini")
    app.add_argument("--provider", choices=["openai", "bedrock"], default="openai")
    app.add_argument("--aws_region", type=str, default="us-east-1", help="Only used with --provider bedrock")
    app.add_argument("--guideline_file", type=str, required=True)
    app.add_argument("--output_dir", type=str, default="outputs_llm")
    app.add_argument("--concurrency", type=int, default=10, help="Parallel API requests. Raise/lower if you hit rate limits.")
    app.add_argument("--limit", type=int, default=None, help="Only process the first N examples per split (for testing).")
    app.set_defaults(func=cmd_apply)

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
