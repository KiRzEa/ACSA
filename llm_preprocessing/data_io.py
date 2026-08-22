#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal block-format reader/writer, shared by all 3 dataset scripts.
Deliberately reimplemented here (not importing train_mtl_acsa_v2) to keep
this LLM-only tooling free of the heavy torch/transformers dependency."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ANN_RE = re.compile(r"\{([^,{}]+),\s*([^{}]+)\}")


@dataclass
class Example:
    sample_id: str
    text: str
    labels: List[Tuple[str, str]]


def load_examples(path: str | Path) -> List[Example]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    examples = []
    for block_idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("#"):
            sample_id = lines[0][1:].strip()
            content = lines[1:]
        else:
            sample_id = str(block_idx)
            content = lines
        ann_idx = next((i for i, l in enumerate(content) if "{" in l and "}" in l), None)
        if ann_idx is None:
            continue
        text = " ".join(content[:ann_idx]).strip()
        ann_text = " ".join(content[ann_idx:]).strip()
        labels = [(c.strip(), s.strip().lower()) for c, s in ANN_RE.findall(ann_text)]
        examples.append(Example(sample_id=sample_id, text=text, labels=labels))
    return examples


def write_examples(path: str | Path, examples: List[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            labels_str = ", ".join(f"{{{c}, {s}}}" for c, s in ex.labels)
            f.write(f"#{ex.sample_id}\n{ex.text}\n{labels_str}\n\n")


def gold_labels_str(labels: List[Tuple[str, str]]) -> str:
    return "\n".join(f"- {c}: {s}" for c, s in labels)
