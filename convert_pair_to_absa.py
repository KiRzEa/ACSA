#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a domain's cleaned Pair-format data (data/Pair/<Domain>/{Train,Dev,
Test}.csv) into the block-format .txt files train_mtl_acsa_v2.py reads
(#id / text / {category, polarity}, ... / blank line), matching the exact
convention already used by Education_ABSA and Hotel_ABSA.

Category tags are written as the domain dict's Vietnamese phrase (from
mapper.DOMAIN_DICTS[domain]), capitalized -- e.g. beauty_dict's 'colour' ->
'màu sắc' -> 'Màu sắc' -- the same convention Education_ABSA/*.txt already
uses (mapper.load_pair_examples returns the dict's English key, not the
phrase, so this script does that key -> phrase -> capitalize step).

Usage:
    python convert_pair_to_absa.py --domain Beauty
    python convert_pair_to_absa.py --domain Beauty --output_dir Beauty_ABSA
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mapper
from llm_preprocessing.data_io import Example, write_examples

# The block format delimits examples by blank lines ("\n\s*\n"), so any
# embedded newline left inside a text field -- common in this raw data,
# which includes copy-pasted multi-line social-media posts -- would corrupt
# block boundaries on re-parse (wrong sample_id, truncated text, or a
# fragment silently dropped for lacking an annotation line). Collapse all
# whitespace runs (including embedded newlines) to a single space so every
# example is strictly one line, matching the invariant every other domain's
# block-format file already relies on.
_WS_RE = re.compile(r"\s+")


def _sanitize_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def convert_split(domain: str, csv_path: Path, txt_path: Path) -> tuple[int, int]:
    examples, skipped = mapper.load_pair_examples(domain, str(csv_path))
    domain_dict = mapper.DOMAIN_DICTS[domain]
    out = []
    for sample_id, text, labels in examples:
        mapped_labels = [(domain_dict[cat].capitalize(), sent) for cat, sent in labels]
        out.append(Example(sample_id=sample_id, text=_sanitize_text(text), labels=mapped_labels))
    write_examples(txt_path, out)
    return len(out), skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=sorted(mapper.DOMAIN_DICTS.keys()))
    p.add_argument("--pair_dir", default=None, help="Defaults to data/Pair/<Domain>")
    p.add_argument("--output_dir", default=None, help="Defaults to <Domain>_ABSA")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    pair_dir = Path(args.pair_dir) if args.pair_dir else root / "data" / "Pair" / args.domain
    output_dir = Path(args.output_dir) if args.output_dir else root / f"{args.domain}_ABSA"

    for split in ("Train", "Dev", "Test"):
        csv_path = pair_dir / f"{split}.csv"
        txt_path = output_dir / f"{split}.txt"
        n, skipped = convert_split(args.domain, csv_path, txt_path)
        print(f"[{args.domain}] {split}: {n} examples written -> {txt_path} (skipped={skipped})")


if __name__ == "__main__":
    main()
