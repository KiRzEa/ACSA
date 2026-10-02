"""Query-paraphrase-robustness eval (experiment (b), see plan file): does CAGE's per-category
prediction change when its query text is reworded (same meaning, different surface form)?

Loads a checkpoint trained by run_cage_checkpoint.sh (outputs/cage_ckpt_<domain>_fixed/seed_42/
best_model.pt) the same way infer_v2.py does -- everything about the run is read back from
checkpoint["args"]/checkpoint["categories"], only --checkpoint is required -- then re-runs test-set
inference 1 + len(paraphrases) times:
  round "original"  uses the model's own encoded category queries, unchanged (sanity check: this
                     arm's per-category F1 should match the checkpoint's own test_metrics.json)
  round "para_1",... every category's query text is swapped for its i-th paraphrase (paraphrases.py)
                     SIMULTANEOUSLY, not one category at a time. This is equivalent to (but far
                     cheaper than) a per-category isolated swap: cross-attention is per-category-row
                     independent (model.py::forward), so each category's score only depends on its
                     own row's query text, never on what any other row's query says -- so swapping
                     every row at once for one round costs one test-set pass instead of K.

Reuses train_mtl_acsa_v2.py's own collect_outputs/compute_metrics and the exact per-category
joint (category, polarity) F1 computed from the gold/pred arrays the same way
run_query_substitution_eval() does (see train_mtl_acsa_v2.py) -- no jsonl round-trip needed.

Writes outputs/cage_ckpt_<domain>_fixed/seed_42/paraphrase_eval.json:
  {"domain", "categories", "rounds": {"original": [f1...], "para_1": [f1...], "para_2": [f1...]}}
(one F1 per category, same order as "categories"). Aggregate across domains with
scripts/paraphrase_stats.py.

Usage: python3 scripts/paraphrase_robustness.py --checkpoint outputs/cage_ckpt_beauty_fixed/seed_42/best_model.pt
       python3 scripts/paraphrase_robustness.py --domain beauty   # shorthand for the path above
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_mtl_acsa_v2 as T  # noqa: E402
from mapper import get_category_descriptions, get_category_names  # noqa: E402
from paraphrases import get_category_paraphrases  # noqa: E402


def _load_checkpoint(checkpoint_path: Path) -> dict:
    # weights_only=False: see infer_v2.py's identical comment -- written by our own
    # save_checkpoint(), not an untrusted source.
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _build_loader(examples, categories, tokenizer, segmenter, max_length, batch_size) -> DataLoader:
    dataset = T.ACSADataset(examples, categories)
    collate = T.make_collate_fn(tokenizer, segmenter, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _per_category_f1(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """gold, pred: [N, K] joint label ids (0=NONE). Same formula as
    run_query_substitution_eval()/category_analysis.py's per_category_f1, over arrays not jsonl."""
    tp = ((pred == gold) & (gold > 0)).sum(0)
    fp = ((pred > 0) & (pred != gold)).sum(0)
    fn = ((gold > 0) & (pred != gold)).sum(0)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 200.0 * tp / np.maximum(denom, 1), 0.0)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    train_args = argparse.Namespace(**checkpoint["args"])
    categories = checkpoint["categories"]
    domain = args.domain or getattr(train_args, "domain", None)
    if not domain:
        raise ValueError("checkpoint has no saved --domain; pass --domain explicitly")
    print(f"Domain: {domain}  Categories: {len(categories)}")

    segmenter = T.build_segmenter(train_args.segmenter)
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name, use_fast=False)

    # Ground truth for what the model actually trained on: the category_texts.json saved next to
    # the checkpoint, not a fresh mapper.py lookup (mapper.py could drift after training -- see
    # infer_v2.py's identical caution).
    saved_texts_path = checkpoint_path.parent / "category_texts.json"
    if saved_texts_path.exists():
        cat_desc = json.loads(saved_texts_path.read_text(encoding="utf-8"))
    elif getattr(train_args, "category_text", "description") == "name":
        cat_desc = get_category_names(categories)
    else:
        cat_desc = get_category_descriptions(categories, domain)
    original_texts = [segmenter(cat_desc[c]) for c in categories]

    paraphrases = get_category_paraphrases(categories, domain)  # {category: [p1, p2]}
    n_variants = len(next(iter(paraphrases.values())))
    variant_texts = [[segmenter(paraphrases[c][i]) for c in categories] for i in range(n_variants)]

    model = T.CategoryConditionedMTL(
        model_name=train_args.model_name,
        tokenizer=tokenizer,
        categories=categories,
        category_texts=original_texts,
        num_attention_heads=train_args.num_attention_heads,
        adapter_dim=train_args.adapter_dim,
        dropout=train_args.dropout,
        gradient_checkpointing=False,
        entity_attribute_heads=getattr(train_args, "entity_attribute_heads", False),
        learned_fusion=getattr(train_args, "learned_fusion", False),
        fusion_gate=getattr(train_args, "fusion_gate", False),
        gate_mode=getattr(train_args, "gate_mode", "soft"),
        category_query=getattr(train_args, "category_query", "text"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Device: {device}")

    test_path = args.data_path or train_args.test_path
    test_examples = T.parse_dataset(test_path)
    print(f"Loaded {len(test_examples)} test examples from {test_path}")
    batch_size = args.batch_size or train_args.eval_batch_size
    loader = _build_loader(test_examples, categories, tokenizer, segmenter, train_args.max_length, batch_size)
    threshold = float(checkpoint["threshold"])

    original_encode = model._encode_category_queries

    def _encode_texts(texts):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=48, return_tensors="pt")
        cat_out = model.encoder(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            return_dict=True,
        ).last_hidden_state
        return model.cat_query_proj(cat_out[:, 0, :])  # identical to _encode_category_queries's text branch

    rounds = {}
    try:
        print("Round: original")
        raw = T.collect_outputs(model, loader, device)
        metrics, pred = T.compute_metrics(raw, threshold)
        gold = raw["joint_labels"].astype(np.int64)
        rounds["original"] = _per_category_f1(gold, pred).tolist()
        print(f"  acsa_f1_micro={metrics['acsa_f1_micro']:.4f} (sanity check vs checkpoint's own test_metrics.json)")

        for i, texts in enumerate(variant_texts, start=1):
            print(f"Round: para_{i}")
            q = _encode_texts(texts).detach()
            model._encode_category_queries = lambda q=q: q
            raw = T.collect_outputs(model, loader, device)
            metrics, pred = T.compute_metrics(raw, threshold)
            gold = raw["joint_labels"].astype(np.int64)
            rounds[f"para_{i}"] = _per_category_f1(gold, pred).tolist()
            print(f"  acsa_f1_micro={metrics['acsa_f1_micro']:.4f}")
    finally:
        model._encode_category_queries = original_encode

    out = {"domain": domain, "categories": list(categories), "rounds": rounds}
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "paraphrase_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    orig = np.array(rounds["original"])
    for i in range(1, n_variants + 1):
        delta = np.array(rounds[f"para_{i}"]) - orig
        print(f"para_{i}: mean per-category F1 delta = {delta.mean():+.2f}  (n={len(delta)} categories)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paraphrase-robustness eval for a trained CAGE checkpoint")
    p.add_argument("--checkpoint", type=str, default=None, help="path to best_model.pt")
    p.add_argument("--domain", type=str, default=None,
                   help="shorthand: resolves to outputs/cage_ckpt_<domain>_fixed/seed_42/best_model.pt "
                        "if --checkpoint is not given; also overrides the checkpoint's saved domain if given")
    p.add_argument("--data_path", type=str, default=None, help="override the test split path saved in the checkpoint's args")
    p.add_argument("--output_dir", type=str, default=None, help="defaults to the checkpoint's own directory")
    p.add_argument("--batch_size", type=int, default=None, help="defaults to the checkpoint's eval_batch_size")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if not args.checkpoint:
        if not args.domain:
            raise SystemExit("pass --checkpoint or --domain")
        args.checkpoint = f"outputs/cage_ckpt_{args.domain}_fixed/seed_42/best_model.pt"
    run(args)
