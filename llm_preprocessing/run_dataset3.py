#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset 3: structured (entity, attribute, sentiment) extraction, run on
Dataset 1's already-cleaned text (not raw). One verification pass against
gold using the exact same P/R/F formula as evaluate_jsonl_per_category --
reports how good zero-shot LLM extraction is, without a full Reflector loop.

    python run_dataset3.py --domain hotel --model gpt-4o-mini \\
        --dataset1_dir outputs_llm/hotel_dataset1
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from gvr_core import run_extractor
from llm_client import get_tracker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import evaluate_jsonl_per_category  # noqa: E402


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / f"{args.domain}_dataset3"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = get_tracker(output_dir / "cost.json")
    dataset1_dir = Path(args.dataset1_dir)

    for split in ("Train", "Dev", "Test"):
        src_path = dataset1_dir / f"{split}.jsonl"
        if not src_path.exists():
            print(f"[{split}] skip -- {src_path} not found (run run_dataset1.py apply first)")
            continue
        records = [json.loads(line) for line in src_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.limit:
            records = records[:args.limit]
        out_path = output_dir / f"{split}.jsonl"

        done_ids = set()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
            print(f"[{split}] Resuming: {len(done_ids)} already done")

        todo = [r for r in records if r["id"] not in done_ids]
        write_lock = threading.Lock()
        failures_path = output_dir / f"{split}_failures.jsonl"
        with out_path.open("a", encoding="utf-8") as f, \
             failures_path.open("a", encoding="utf-8") as ferr, \
             tqdm(total=len(todo), desc=f"{split}", unit="ex") as bar:

            def process(rec):
                try:
                    extractions = run_extractor(rec["text"], args.domain, args.model, tracker, args.provider, args.aws_region)
                    out_record = {
                        "id": rec["id"], "text": rec["text"], "gold": rec["gold"],
                        "prediction": [{"category": c, "sentiment": s} for c, s in extractions],
                    }
                    with write_lock:
                        f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:
                    with write_lock:
                        ferr.write(json.dumps({"id": rec["id"], "error": str(e)}, ensure_ascii=False) + "\n")
                        ferr.flush()
                with write_lock:
                    bar.update(1)
                    bar.set_postfix(cost=f"${tracker.totals['cost_usd']:.4f}")

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(process, rec) for rec in todo]
                for fut in as_completed(futures):
                    fut.result()

        if failures_path.exists() and failures_path.stat().st_size > 0:
            n_failed = sum(1 for _ in failures_path.read_text(encoding="utf-8").splitlines() if _.strip())
            print(f"[{split}] {n_failed} examples failed, logged to {failures_path} -- re-run to retry them")
        print(f"[{split}] done -> {out_path}")

        print(f"[{split}] verify (LLM extraction vs gold):")
        evaluate_jsonl_per_category(out_path)

    print(f"Dataset 3 TOTAL cost for {args.domain}: {tracker.summary()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["hotel", "restaurant"], required=True)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--provider", choices=["openai", "bedrock"], default="openai")
    p.add_argument("--aws_region", type=str, default="us-east-1", help="Only used with --provider bedrock")
    p.add_argument("--dataset1_dir", type=str, required=True, help="Path to <domain>_dataset1/ from run_dataset1.py apply")
    p.add_argument("--output_dir", type=str, default="outputs_llm")
    p.add_argument("--concurrency", type=int, default=10, help="Parallel API requests. Raise/lower if you hit rate limits.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N records per split (for testing).")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
