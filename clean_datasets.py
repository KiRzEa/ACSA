#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce cleaned copies of the raw Hotel_ABSA / Res_ABSA datasets using
ABSA_LLMs' clean_doc() preprocessing (normalize slang/emoji/diacritics,
strip punctuation, price/number normalization, optional pyvi segmentation).
Only the text line is transformed -- id and {category, sentiment} label
lines are copied through unchanged, so labels stay perfectly aligned.

clean_doc(word_segment=True) already runs pyvi internally, so the output
is pre-segmented: train on these files with --segmenter none (not pyvi),
otherwise the text gets segmented twice.

Usage:
    python clean_datasets.py --domain hotel
    python clean_datasets.py --domain restaurant
    python clean_datasets.py --domain both
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from preprocessing import clean_doc

ANN_RE = re.compile(r"\{[^,{}]+,\s*[^{}]+\}")

DOMAIN_ROOTS = {
    "hotel": "Hotel_ABSA",
    "restaurant": "Res_ABSA",
}


def clean_split(src_path: Path, dst_path: Path, word_segment: bool) -> int:
    raw = src_path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst_path.open("w", encoding="utf-8") as f:
        for block_idx, block in enumerate(blocks, start=1):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if lines[0].startswith("#"):
                sample_id_line = lines[0]
                content = lines[1:]
            else:
                sample_id_line = f"#{block_idx}"
                content = lines

            ann_idx = next(
                (i for i, line in enumerate(content) if "{" in line and "}" in line),
                None,
            )
            if ann_idx is None:
                raise ValueError(f"No annotation line in {src_path}, block {block_idx}")

            text = " ".join(content[:ann_idx]).strip()
            ann_text = " ".join(content[ann_idx:]).strip()

            cleaned_text = clean_doc(text, word_segment=word_segment, lower_case=False, max_length=512)
            if not cleaned_text:
                raise ValueError(
                    f"clean_doc produced empty text for {src_path}, block {block_idx}: "
                    f"original={text!r}"
                )

            f.write(sample_id_line + "\n")
            f.write(cleaned_text + "\n")
            f.write(ann_text + "\n\n")
            n += 1
    return n


def run(domain_key: str, word_segment: bool) -> None:
    root = Path(__file__).resolve().parent
    src_root = root / DOMAIN_ROOTS[domain_key]
    dst_root = root / f"{DOMAIN_ROOTS[domain_key]}_clean"
    for split in ("Train", "Dev", "Test"):
        src = src_root / f"{split}.txt"
        dst = dst_root / f"{split}.txt"
        n = clean_split(src, dst, word_segment=word_segment)
        print(f"[{domain_key}] {split}: {n} examples -> {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["hotel", "restaurant", "both"], required=True)
    p.add_argument(
        "--no_word_segment", action="store_true",
        help="Skip pyvi segmentation inside clean_doc (leave raw syllables); "
        "use this only if you plan to run --segmenter pyvi at train time instead.",
    )
    args = p.parse_args()

    domains = ["hotel", "restaurant"] if args.domain == "both" else [args.domain]
    for d in domains:
        run(d, word_segment=not args.no_word_segment)
