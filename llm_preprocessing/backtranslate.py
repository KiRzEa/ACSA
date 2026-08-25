#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Back-translation augmentation (vi -> en -> vi) via a local HF MT model.
Kept separate from the rest of llm_preprocessing/ (which deliberately avoids
torch/transformers, see data_io.py's docstring) -- torch is lazy-imported
inside load_model() so scripts that only use the LLM augmentation path never
need it installed.

Greedy round-trip translation just reconstructs the original sentence almost
verbatim, which defeats the point of augmentation -- sampling on both legs
is what actually produces distinct paraphrase candidates."""

from __future__ import annotations

from typing import List

_model = None
_tokenizer = None
_model_name = None

# NLLB-200 language codes (not ISO 639-1 -- this exact format is required).
_LANG_CODES = {"vi": "vie_Latn", "en": "eng_Latn"}


def load_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """Loads once per process; ~2.4GB download on first use."""
    global _model, _tokenizer, _model_name
    if _model is not None and _model_name == model_name:
        return _model, _tokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _model.eval()
    _model_name = model_name
    return _model, _tokenizer


def _translate(texts: List[str], src: str, tgt: str, model, tokenizer, sample: bool) -> List[str]:
    import torch

    tokenizer.src_lang = _LANG_CODES[src]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(_LANG_CODES[tgt])
    gen_kwargs = dict(forced_bos_token_id=forced_bos_token_id, max_length=128)
    if sample:
        gen_kwargs.update(do_sample=True, temperature=1.0, top_p=0.92)
    else:
        gen_kwargs.update(num_beams=4)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def paraphrase(seed_text: str, n: int, model_name: str = "facebook/nllb-200-distilled-600M") -> List[str]:
    """vi -> en (greedy, one canonical translation) -> vi (sampled n times)
    -- sampling only on the return leg keeps the English pivot stable while
    still producing n distinct Vietnamese surface forms."""
    model, tokenizer = load_model(model_name)
    [english] = _translate([seed_text], "vi", "en", model, tokenizer, sample=False)
    candidates = []
    for _ in range(n):
        [back] = _translate([english], "en", "vi", model, tokenizer, sample=True)
        candidates.append(back)
    return candidates


if __name__ == "__main__":
    import sys

    samples = [
        "Phòng sạch sẽ, nhân viên thân thiện nhưng giá hơi cao so với mặt bằng chung.",
        "Bữa sáng khá đơn điệu, ít món để lựa chọn.",
        "Vị trí khách sạn rất thuận tiện, gần trung tâm và nhiều quán ăn ngon.",
    ]
    mn = sys.argv[1] if len(sys.argv) > 1 else "facebook/nllb-200-distilled-600M"
    for s in samples:
        print(f"\nSEED: {s}")
        for i, cand in enumerate(paraphrase(s, n=3, model_name=mn), start=1):
            print(f"  [{i}] {cand}")
