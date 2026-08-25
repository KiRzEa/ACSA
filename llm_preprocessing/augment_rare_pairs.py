#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted data augmentation for rare (category, polarity) pairs. Not a
blanket dataset augmentation -- only tops up pairs under --rare_threshold
training examples, via LLM paraphrase generation and/or back-translation.

Every candidate sentence (from either method) is labeled by re-running
run_extractor() on the candidate text itself, not by copying the seed's
label set -- accepted only if the target (category, polarity) shows up in
that fresh extraction. Same self-consistency pattern gvr_core.run_verifier()
already uses, doing QA and labeling in one call.

Usage:
    python augment_rare_pairs.py --domain hotel \
        --train_path ../Hotel_ABSA/Train.txt \
        --output_dir ../outputs_llm/hotel_augment \
        --methods both

Then merge (never overwrites the original):
    cat Hotel_ABSA/Train.txt outputs_llm/hotel_augment/augmented_examples.txt \
        > Hotel_ABSA/Train_augmented.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from data_io import Example, load_examples
from domain_context import category_list
from gvr_core import run_extractor, run_paraphrase_generator
from llm_client import get_tracker

logger = logging.getLogger("augment_rare_pairs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

Pair = Tuple[str, str]


def find_rare_pairs(examples: List[Example], rare_threshold: int, target_count: int) -> Dict[Pair, int]:
    counts: Counter = Counter()
    for ex in examples:
        for cat, pol in ex.labels:
            counts[(cat, pol)] += 1
    return {
        pair: target_count - count
        for pair, count in counts.items()
        if count < rare_threshold and target_count > count
    }


def seeds_for_pair(examples: List[Example], pair: Pair, max_seeds: int = 3) -> List[Example]:
    matches = [ex for ex in examples if pair in ex.labels]
    random.shuffle(matches)
    return matches[:max_seeds]


def generate_llm_candidates(
    seed: Example, category: str, polarity: str, domain: str, n: int,
    model: str, tracker, provider: str,
) -> List[str]:
    try:
        return run_paraphrase_generator(seed.text, category, polarity, domain, n, model, tracker, provider)
    except Exception as e:
        logger.warning("LLM generation failed for %s/%s: %s", category, polarity, e)
        return []


def generate_backtranslation_candidates(seed: Example, n: int, mt_model: str) -> List[str]:
    import backtranslate

    try:
        return backtranslate.paraphrase(seed.text, n=n, model_name=mt_model)
    except Exception as e:
        logger.warning("Back-translation failed for seed %r: %s", seed.sample_id, e)
        return []


def label_and_verify(
    candidate_text: str, target_pair: Pair, domain: str, model: str, tracker, provider: str,
) -> List[Pair] | None:
    extraction = run_extractor(candidate_text, domain, model, tracker, provider=provider)
    if target_pair in extraction:
        return extraction
    return None


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_categories = set(category_list(args.domain))

    examples = load_examples(args.train_path)
    all_pairs = set((c, p) for ex in examples for c, p in ex.labels)
    rare_pairs = find_rare_pairs(examples, args.rare_threshold, args.target_count)
    logger.info(
        "%d/%d unique (category,polarity) pairs need topping up to target_count=%d (total deficit=%d new examples)",
        len(rare_pairs), len(all_pairs), args.target_count, sum(rare_pairs.values()),
    )

    gen_tracker = get_tracker(output_dir / "cost_augment.json")
    rejects_path = output_dir / "rejects.jsonl"
    rejects_f = rejects_path.open("w", encoding="utf-8")

    # Written incrementally (flushed after every accepted example), not
    # batched to the end -- a run over dozens of rare pairs takes hours, and
    # losing every accepted example to a crash/interrupt because nothing hit
    # disk until the very last pair finished is a real, avoidable loss.
    augmented_path = output_dir / "augmented_examples.txt"
    augmented_f = augmented_path.open("w", encoding="utf-8")

    def _append_accepted(ex: Example) -> None:
        labels_str = ", ".join(f"{{{c}, {s}}}" for c, s in ex.labels)
        augmented_f.write(f"#{ex.sample_id}\n{ex.text}\n{labels_str}\n\n")
        augmented_f.flush()

    n_accepted = 0
    fill_summary = []
    n_llm = n_bt = 0

    methods = ["llm", "backtranslate"] if args.methods == "both" else [args.methods]

    for (category, polarity), deficit in tqdm(sorted(rare_pairs.items()), desc="rare pairs"):
        seeds = seeds_for_pair(examples, (category, polarity))
        if not seeds:
            logger.warning("No seed example found for %s/%s -- skipping (shouldn't happen)", category, polarity)
            continue

        filled = 0
        attempts = 0
        max_attempts = deficit * args.max_attempts_per_pair
        method_cycle = 0

        while filled < deficit and attempts < max_attempts:
            seed = seeds[attempts % len(seeds)]
            method = methods[method_cycle % len(methods)]
            method_cycle += 1
            batch_n = min(args.candidates_per_call, deficit - filled + 2)

            if method == "llm":
                raw_candidates = generate_llm_candidates(
                    seed, category, polarity, args.domain, batch_n, args.model, gen_tracker, args.provider
                )
            else:
                raw_candidates = generate_backtranslation_candidates(seed, batch_n, args.mt_model)
            attempts += 1

            if not raw_candidates:
                continue

            for cand_text in raw_candidates:
                if filled >= deficit:
                    break
                if "#" in cand_text:
                    # Real Vietnamese review text never contains "#" -- a category
                    # code (ENTITY#ATTRIBUTE) leaking into the generated sentence
                    # itself (observed in practice despite the prompt's explicit
                    # instruction not to) is a generation defect, not something
                    # worth sending to the extractor at all.
                    rejects_f.write(json.dumps({
                        "category": category, "polarity": polarity, "method": method,
                        "seed_id": seed.sample_id, "candidate_text": cand_text,
                        "reason": "category_code_leaked_into_text",
                    }, ensure_ascii=False) + "\n")
                    continue
                labels = label_and_verify(cand_text, (category, polarity), args.domain, args.model, gen_tracker, args.provider)
                if labels is None:
                    rejects_f.write(json.dumps({
                        "category": category, "polarity": polarity, "method": method,
                        "seed_id": seed.sample_id, "candidate_text": cand_text,
                        "reason": "target_pair_not_in_reextraction",
                    }, ensure_ascii=False) + "\n")
                    continue
                # dict.fromkeys dedupes while preserving order -- the extractor
                # occasionally returns the same (category, sentiment) pair twice
                # in one JSON response (observed in ~2% of accepted examples).
                labels = list(dict.fromkeys((c, p) for c, p in labels if c in valid_categories))
                if not labels:
                    continue
                idx = n_llm if method == "llm" else n_bt
                prefix = "aug_llm" if method == "llm" else "aug_bt"
                _append_accepted(Example(sample_id=f"{prefix}_{idx}", text=cand_text, labels=labels))
                n_accepted += 1
                if method == "llm":
                    n_llm += 1
                else:
                    n_bt += 1
                filled += 1

        fill_summary.append((category, polarity, deficit, filled, attempts))
        logger.info(
            "[%s/%s] filled %d/%d (running total: %d accepted, cost so far: %s)",
            category, polarity, filled, deficit, n_accepted, gen_tracker.summary(),
        )

    rejects_f.close()
    augmented_f.close()

    logger.info("Wrote %d accepted examples (%d llm, %d backtranslate) to %s",
                n_accepted, n_llm, n_bt, augmented_path)
    logger.info("Rejected candidates logged to %s", rejects_path)
    logger.info("Cost: %s", gen_tracker.summary())

    print("\n=== per-pair fill summary ===")
    print(f"{'category':<32}{'polarity':<10}{'requested':>10}{'filled':>8}{'attempts':>10}")
    for category, polarity, deficit, filled, attempts in fill_summary:
        flag = "" if filled >= deficit else "  <-- INCOMPLETE"
        print(f"{category:<32}{polarity:<10}{deficit:>10}{filled:>8}{attempts:>10}{flag}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=["hotel", "restaurant"], required=True)
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--rare_threshold", type=int, default=20)
    p.add_argument("--target_count", type=int, default=15)
    p.add_argument(
        "--methods", choices=["llm", "backtranslate", "both"], default="llm",
        help="Default llm-only: a real run showed back-translation (NLLB-200 vi<->en round-trip) "
        "produces disfluent/broken Vietnamese on a sizeable fraction of outputs (e.g. 'Xếp này được "
        "thuê vào 580 nghìn đồng đồng/ngày') that the extractor-based verifier still lets through "
        "since it only checks label presence, not fluency. Pass --methods both/backtranslate to opt "
        "back in if you add a fluency check.",
    )
    p.add_argument("--model", type=str, default="gpt-5-nano")
    p.add_argument("--provider", choices=["openai", "bedrock"], default="openai")
    p.add_argument("--mt_model", type=str, default="facebook/nllb-200-distilled-600M")
    p.add_argument("--max_attempts_per_pair", type=int, default=4,
                    help="Multiplier on deficit -- stop trying a pair after deficit*this many generation calls.")
    p.add_argument("--candidates_per_call", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    run(args)
