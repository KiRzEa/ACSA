#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generator / Verifier / Reflector primitives -- one call each, no looping
logic here (that lives in run_dataset1.py). Kept separate so Dataset 2/3
scripts can reuse run_generator/run_extractor without pulling in the loop."""

from __future__ import annotations

from typing import List, Tuple

from llm_client import call_json, CostTracker
from domain_context import DOMAIN_LABELS, category_context_block
import prompts as P


def run_generator(
    text: str, guideline: str, domain: str, model: str, tracker: CostTracker,
    provider: str = "openai", aws_region: str = "us-east-1",
) -> str:
    system = P.GENERATOR_SYSTEM.format(guideline=guideline)
    user = P.GENERATOR_USER.format(text=text)
    result = call_json(system, user, tracker, model=model, provider=provider, aws_region=aws_region)
    return result.get("cleaned_text", text)


def run_verifier(
    original_text: str, cleaned_text: str, labels: List[Tuple[str, str]], domain: str, model: str, tracker: CostTracker,
    provider: str = "openai", aws_region: str = "us-east-1",
) -> Tuple[bool, list]:
    """Extraction-based self-consistency check: compare extraction(cleaned)
    against extraction(ORIGINAL) -- not against gold directly.

    History: direct LLM judgment (cleaned vs gold) hallucinated badly
    (claimed evidence missing from text containing it verbatim, repeatably).
    Switched to extraction(cleaned) vs gold instead -- but that conflated
    two different things: whether cleaning lost information, and whether
    the extractor's own recall is good enough to match gold at all. Verified
    directly: extraction on the RAW, untouched original text returns just as
    empty on implicit/subtle sentences (e.g. "Tuong lai toi se tiep tuc ghe
    lai day" -> [] even pre-cleaning) -- so most "missing" fails were really
    just the extractor's baseline recall limit, unrelated to the Generator.

    Comparing cleaned against original instead isolates exactly what we
    actually want to verify: did rewriting change what's extractable,
    holding the extractor's own accuracy constant as a control."""
    extracted_before = dict(run_extractor(original_text, domain, model, tracker, provider, aws_region))
    extracted_after = dict(run_extractor(cleaned_text, domain, model, tracker, provider, aws_region))
    issues = []
    for cat, sent in extracted_before.items():
        if cat not in extracted_after:
            issues.append({
                "category": cat, "gold_sentiment": sent, "problem": "missing",
                "detail": f"Present in extraction(original)={sent!r} but absent from extraction(cleaned)",
            })
        elif extracted_after[cat] != sent:
            issues.append({
                "category": cat, "gold_sentiment": sent, "problem": "sentiment_changed",
                "detail": f"extraction(original)={sent!r} vs extraction(cleaned)={extracted_after[cat]!r}",
            })
    for cat, sent in extracted_after.items():
        if cat not in extracted_before:
            issues.append({
                "category": cat, "gold_sentiment": sent, "problem": "hallucinated",
                "detail": f"extraction(cleaned) introduces {cat}={sent!r} not present in extraction(original)",
            })
    return len(issues) == 0, issues


def run_reflector(
    guideline: str, failure_examples: str, n_fails: int, n_total: int, model: str, tracker: CostTracker,
    provider: str = "openai", aws_region: str = "us-east-1",
) -> Tuple[str, str]:
    system = P.REFLECTOR_SYSTEM
    user = P.REFLECTOR_USER.format(
        guideline=guideline, failure_examples=failure_examples, n_fails=n_fails, n_total=n_total
    )
    result = call_json(system, user, tracker, model=model, provider=provider, aws_region=aws_region)
    return result.get("updated_guideline", guideline), result.get("summary_of_changes", "")


def run_static_cleaner(
    text: str, domain: str, model: str, tracker: CostTracker,
    provider: str = "openai", aws_region: str = "us-east-1",
) -> str:
    system = P.STATIC_CLEANER_SYSTEM.format(
        domain_label=DOMAIN_LABELS[domain], category_context=category_context_block(domain)
    )
    user = P.STATIC_CLEANER_USER.format(text=text)
    result = call_json(system, user, tracker, model=model, provider=provider, aws_region=aws_region)
    return result.get("cleaned_text", text)


def run_extractor(
    text: str, domain: str, model: str, tracker: CostTracker,
    provider: str = "openai", aws_region: str = "us-east-1",
) -> List[Tuple[str, str]]:
    system = P.EXTRACTOR_SYSTEM.format(
        domain_label=DOMAIN_LABELS[domain], category_context=category_context_block(domain)
    )
    user = P.EXTRACTOR_USER.format(text=text)
    result = call_json(system, user, tracker, model=model, provider=provider, aws_region=aws_region)
    return [(e["category"], e["sentiment"]) for e in result.get("extractions", []) if "category" in e and "sentiment" in e]
