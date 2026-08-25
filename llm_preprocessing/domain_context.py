#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-aware context block for prompts, reusing mapper.py's category
descriptions so the Generator/Verifier/Extractor know what aspects this
domain actually cares about."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mapper  # noqa: E402

DOMAIN_LABELS = {"hotel": "Hotel", "restaurant": "Restaurant"}


def category_context_block(domain: str) -> str:
    """A '<CATEGORY>: <mô tả>' block for every category in this domain,
    so the LLM knows the exact category vocabulary and what each one
    means -- not asked to invent its own aspect taxonomy."""
    domain_key = DOMAIN_LABELS[domain]
    descriptions = mapper.CATEGORY_DESCRIPTIONS[domain_key]
    lines = [f"{cat}: {desc}" for cat, desc in descriptions.items()]
    return "\n".join(lines)


def category_list(domain: str) -> list[str]:
    domain_key = DOMAIN_LABELS[domain]
    return sorted(mapper.CATEGORY_DESCRIPTIONS[domain_key].keys())


def category_description(domain: str, category: str) -> str:
    domain_key = DOMAIN_LABELS[domain]
    return mapper.CATEGORY_DESCRIPTIONS[domain_key].get(category, category)
