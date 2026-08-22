#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset 2: raw text -> static system prompt, single pass, no verification
loop. Ablation baseline against Dataset 1 -- does the Generator/Verifier/
Reflector refinement actually earn its cost over just prompting once?

    python run_dataset2.py --domain hotel --model gpt-4o-mini
    python run_dataset2.py --domain hotel --model amazon.nova-micro-v1:0 --provider bedrock
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from data_io import Example, load_examples
from gvr_core import run_static_cleaner
from llm_client import get_tracker

RAW_ROOT = {"hotel": "Hotel_ABSA", "restaurant": "Res_ABSA"}


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / f"{args.domain}_dataset2"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = get_tracker(output_dir / "cost.json")

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
        with out_path.open("a", encoding="utf-8") as f, \
             failures_path.open("a", encoding="utf-8") as ferr, \
             tqdm(total=len(todo), desc=f"{split}", unit="ex") as bar:

            def process(ex: Example):
                try:
                    cleaned = run_static_cleaner(ex.text, args.domain, args.model, tracker, args.provider, args.aws_region)
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
                    bar.set_postfix(cost=f"${tracker.totals['cost_usd']:.4f}")

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(process, ex) for ex in todo]
                for fut in as_completed(futures):
                    fut.result()

        if failures_path.exists() and failures_path.stat().st_size > 0:
            n_failed = sum(1 for _ in failures_path.read_text(encoding="utf-8").splitlines() if _.strip())
            print(f"[{split}] {n_failed} examples failed, logged to {failures_path} -- re-run to retry them")
        print(f"[{split}] done -> {out_path}")

    print(f"Dataset 2 TOTAL cost for {args.domain}: {tracker.summary()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["hotel", "restaurant"], required=True)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--provider", choices=["openai", "bedrock"], default="openai")
    p.add_argument("--aws_region", type=str, default="us-east-1", help="Only used with --provider bedrock")
    p.add_argument("--output_dir", type=str, default="outputs_llm")
    p.add_argument("--concurrency", type=int, default=10, help="Parallel API requests. Raise/lower if you hit rate limits.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N examples per split (for testing).")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
