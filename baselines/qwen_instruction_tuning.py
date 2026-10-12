#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruction-tuned LLM baseline: Qwen2.5-3B-Instruct fine-tuned with LoRA through Unsloth (the
"NL/Code Instruction-Vi/En" rows of the main results table, replacing the ViT5-base backbone of
t5_instruction_tuning.py).

Only the backbone changes. The four prompt variants, the generation targets, the output parser and the
category canonicalization are imported unchanged from t5_instruction_tuning.py, so the two backbones are
compared on identical inputs and identical scoring. The prompt is sent as the user turn of the Qwen chat
template and the target is the assistant turn; the loss covers the target tokens only.

Training: 4-bit base weights (QLoRA) with LoRA adapters on every attention and MLP projection, AdamW (8-bit)
with a linear schedule and 10% warmup, up to 3 epochs; the dev loss is computed after every epoch, training stops
when it does not improve for --patience epochs, and the adapters of the epoch with the lowest dev loss are kept
(in CPU RAM; nothing is written to disk). Test predictions use greedy decoding.

Backends:
  unsloth  (default when CUDA is available; Kaggle T4: fp16, one GPU per process -- set CUDA_VISIBLE_DEVICES)
  hf       (plain transformers, optional peft LoRA; for CPU smoke tests of the data/training/decoding path)

Outputs in --output_dir (same layout as t5_instruction_tuning.py, so the statistics scripts read both):
  test_predictions.jsonl   parsed (category, sentiment) pairs, scored by common.micro_prf
  test_generations.jsonl   raw model text, so the output can be re-parsed without retraining
  multi_seed_summary.json  test P/R/F1 per seed
  run_config.json          everything needed to report the run: command, code commit, hardware, package versions,
                           model and revision, quantization, LoRA, every TrainingArguments value, decoding, prompt, data
  history.json             per seed: loss curves, dev loss per epoch, stopping epoch, timings, memory, output statistics
scripts/qwen_run_report.py aggregates these files over all runs.

Example (one job of the Kaggle queue):
    CUDA_VISIBLE_DEVICES=0 python3 baselines/qwen_instruction_tuning.py --domain hotel --format nl --lang vi \\
        --train_path Hotel_ABSA/Train.txt --dev_path Hotel_ABSA/Dev.txt --test_path Hotel_ABSA/Test.txt \\
        --output_dir outputs/qwen_hotel_nl_vi/seed_42 --seeds 42
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU per process; the Kaggle queue sets it explicitly
import torch  # noqa: E402

# Unsloth patches transformers at import time, so it must be imported before anything imports transformers
# (t5_instruction_tuning below does). Only attempted when a GPU is present and the hf backend is not forced.
_WANT_HF = "--backend" in sys.argv and sys.argv[sys.argv.index("--backend") + 1:sys.argv.index("--backend") + 2] == ["hf"]
if torch.cuda.is_available() and not _WANT_HF:
    try:
        import unsloth  # noqa: F401
    except ImportError:
        pass

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import enum  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SENTIMENTS, Example, infer_categories, load_examples, micro_prf, write_metrics, write_predictions  # noqa: E402
from t5_instruction_tuning import build_prompt, build_target, canonicalize_category, parse_prediction  # noqa: E402

logger = logging.getLogger("qwen_instruction_tuning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

DEFAULT_MODEL = {"unsloth": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "hf": "Qwen/Qwen2.5-3B-Instruct"}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chat_prompt(tokenizer, args, review: str) -> str:
    """The instruction of t5_instruction_tuning.build_prompt as the user turn, ending with the assistant header."""
    return tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(args.format, args.lang, args.domain, review)}],
                                         tokenize=False, add_generation_prompt=True)


class ChatSFTDataset(Dataset):
    """input_ids = chat prompt + target + EOS; labels mask the prompt with -100 (loss on the answer only)."""

    def __init__(self, examples: List[Example], tokenizer, args):
        self.items = []
        truncated = 0
        for ex in examples:
            p = tokenizer(chat_prompt(tokenizer, args, ex.text), add_special_tokens=False)["input_ids"]
            t = tokenizer(build_target(args.format, args.lang, ex.labels) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            if len(p) + len(t) > args.max_seq_length:  # keep the whole answer, cut the prompt from the left
                p = p[len(p) + len(t) - args.max_seq_length:]
                truncated += 1
            self.items.append({"input_ids": p + t, "labels": [-100] * len(p) + t})
        self.truncated = truncated
        if truncated:
            logger.warning("%d of %d examples exceeded max_seq_length=%d; their prompts were cut", truncated, len(examples), args.max_seq_length)

    def length_stats(self) -> dict:
        lens = sorted(len(x["input_ids"]) for x in self.items)
        answer = [sum(1 for t in x["labels"] if t != -100) for x in self.items]
        return {"n": len(lens), "tokens": sum(lens), "mean_tokens": round(sum(lens) / max(1, len(lens)), 1),
                "p95_tokens": lens[int(0.95 * (len(lens) - 1))] if lens else 0, "max_tokens": lens[-1] if lens else 0,
                "mean_answer_tokens": round(sum(answer) / max(1, len(answer)), 1), "truncated": self.truncated}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):  # right padding for training
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class BestAdapterTracker(TrainerCallback):
    """Keeps the trainable (LoRA) weights of the epoch with the lowest dev loss in CPU RAM, like
    t5_instruction_tuning.BestModelTracker; Kaggle disk is never touched."""

    def __init__(self):
        self.t0 = time.time()
        self.best_loss = None
        self.best_epoch = None
        self.best_state = None
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not metrics or model is None or "eval_loss" not in metrics:
            return
        loss = float(metrics["eval_loss"])
        self.history.append({"epoch": round(float(state.epoch or 0), 2), "dev_loss": loss, "elapsed_s": round(time.time() - self.t0, 1)})
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss, self.best_epoch = loss, round(float(state.epoch or 0), 2)
            self.best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    def restore(self, model) -> None:
        if self.best_state is None:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.best_state:
                    p.copy_(self.best_state[n].to(p.device, dtype=p.dtype))


def load_model(args, seed: int):
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model_name, max_seq_length=args.max_seq_length,
                                                             dtype=None, load_in_4bit=args.load_in_4bit)
        model = FastLanguageModel.get_peft_model(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                                                 target_modules=LORA_TARGETS, use_gradient_checkpointing="unsloth", random_state=seed)
        return model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)
    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, device_map={"": 0} if cuda else None)
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):  # tiny test models with a borrowed tokenizer
        model.resize_token_embeddings(len(tokenizer))
    try:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    except ImportError:
        logger.warning("peft not installed: hf backend trains all weights (smoke tests only)")
    if cuda:  # GPU fallback without Unsloth: 16-bit frozen weights, fp32 trainable adapters, checkpointed activations
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, examples: List[Example], args, categories: List[str]) -> Tuple[List[List[Tuple[str, str]]], List[str]]:
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    prompts = [chat_prompt(tokenizer, args, ex.text) for ex in examples]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch: less padding
    raw: List[str] = [""] * len(prompts)
    for start in range(0, len(order), args.eval_batch_size):
        idx = order[start:start + args.eval_batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            raw[i] = t.strip()
    preds = [[(canonicalize_category(c, categories), s) for c, s in parse_prediction(args.format, args.lang, t)] for t in raw]
    return preds, raw


def training_arguments(args, n_train: int, seed: int, output_dir: Path, gpu: bool, bf16: bool) -> TrainingArguments:
    """TrainingArguments for transformers 4.x and 5.x. Version 5 dropped warmup_ratio (the warmup is then given in
    steps) and may drop other options; anything this version does not know is left out and reported."""
    total_steps = args.max_steps or math.ceil(n_train / (args.batch_size * args.grad_accum)) * args.epochs
    kw = dict(output_dir=str(output_dir / "trainer_state"), num_train_epochs=args.epochs,
              per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
              per_device_eval_batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
              lr_scheduler_type="linear", optim="adamw_8bit" if args.backend == "unsloth" else "adamw_torch",
              eval_strategy="epoch", save_strategy="no", metric_for_best_model="eval_loss", greater_is_better=False,
              prediction_loss_only=True, group_by_length=True, logging_steps=25, disable_tqdm=True, report_to=[],
              seed=seed, data_seed=seed, bf16=bf16, fp16=gpu and not bf16, use_cpu=not torch.cuda.is_available(),
              remove_unused_columns=False, max_steps=args.max_steps if args.max_steps else -1)
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "warmup_ratio" in fields:
        kw["warmup_ratio"] = args.warmup_ratio
    else:
        kw["warmup_steps"] = math.ceil(args.warmup_ratio * total_steps)
    if "group_by_length" not in fields and "train_sampling_strategy" in fields:  # 5.x name of the same sampler
        kw.pop("group_by_length")
        kw["train_sampling_strategy"] = "group_by_length"
    dropped = sorted(k for k in kw if k not in fields)
    if dropped:
        logger.warning("this transformers version has no TrainingArguments %s; left out", dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in fields})


def jsonable(x):
    """Plain JSON for configs that hold sets, enums, dtypes or paths."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(jsonable(v) for v in x)
    if isinstance(x, enum.Enum):
        return x.value
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def package_versions() -> dict:
    from importlib import metadata
    out = {"python": sys.version.split()[0], "cuda_runtime": torch.version.cuda}
    for pkg in ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "unsloth", "unsloth_zoo", "xformers", "triton"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def hardware() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    prop = torch.cuda.get_device_properties(0)
    return {"device": prop.name, "gpu_memory_gb": round(prop.total_memory / 2**30, 2), "visible_gpus": torch.cuda.device_count(),
            "compute_capability": f"{prop.major}.{prop.minor}", "bf16_supported": torch.cuda.is_bf16_supported(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def code_version() -> dict:
    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        return {"git_commit": sha or None, "git_dirty": bool(dirty)}
    except Exception:  # not a git checkout
        return {"git_commit": None, "git_dirty": None}


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_record(model, args) -> dict:
    cfg = getattr(model, "config", None)
    revision = getattr(cfg, "_commit_hash", None)
    if revision is None and not Path(args.model_name).exists():  # ask the Hub which revision was downloaded
        try:
            from huggingface_hub import HfApi
            revision = HfApi().model_info(args.model_name, timeout=20).sha
        except Exception:
            pass
    rec = {"model_name": args.model_name, "backend": args.backend, "model_revision": revision,
           "load_in_4bit": bool(args.load_in_4bit), "dtype": str(getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None))}
    quant = getattr(cfg, "quantization_config", None)
    if quant is not None:
        rec["quantization_config"] = jsonable(quant.to_dict() if hasattr(quant, "to_dict") else quant)
    peft_cfg = getattr(model, "peft_config", None)
    if peft_cfg:
        rec["lora_config"] = jsonable(next(iter(peft_cfg.values())).to_dict())
    if hasattr(model, "get_nb_trainable_parameters"):
        rec["trainable_parameters"], rec["total_parameters"] = model.get_nb_trainable_parameters()
    else:
        rec["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rec["total_parameters"] = sum(p.numel() for p in model.parameters())
    rec["gradient_checkpointing"] = "unsloth" if args.backend == "unsloth" else torch.cuda.is_available()
    return rec


def output_stats(raw: List[str], examples: List[Example], args, categories: List[str]) -> dict:
    """How well the generations follow the target format (reported with the results)."""
    pairs = unknown = bad_sent = fuzzy = no_pairs = empty = dup = 0
    for t in raw:
        parsed = parse_prediction(args.format, args.lang, t)
        empty += not t.strip()
        no_pairs += not parsed
        canon = [(canonicalize_category(c, categories), s) for c, s in parsed]
        dup += len(canon) - len(set(canon))
        for (c, s), (cc, _) in zip(parsed, canon):
            pairs += 1
            unknown += cc not in categories
            fuzzy += cc in categories and cc != c.strip()
            bad_sent += s not in SENTIMENTS
    return {"reviews": len(raw), "gold_reviews_without_labels": sum(1 for ex in examples if not ex.labels),
            "empty_generations": empty, "generations_without_pairs": no_pairs, "pairs": pairs,
            "pairs_unknown_category": unknown, "pairs_fuzzy_matched_category": fuzzy, "pairs_invalid_sentiment": bad_sent,
            "duplicate_pairs": dup}


def run(args) -> None:
    train, dev, test = load_examples(args.train_path), load_examples(args.dev_path), load_examples(args.test_path)
    categories = infer_categories(train, dev, test)
    if args.limit:  # smoke tests
        train, dev, test = train[:args.limit], dev[:max(4, args.limit // 4)], test[:max(4, args.limit // 4)]
    logger.info("Loaded %d train / %d dev / %d test (domain=%s, format=%s, lang=%s, backend=%s)",
                len(train), len(dev), len(test), args.domain, args.format, args.lang, args.backend)
    output_dir = Path(args.output_dir)
    per_seed, run_config = [], {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        set_seed(seed)
        t0 = time.time()
        model, tokenizer = load_model(args, seed)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not args.max_new_tokens:  # longest training answer + margin
            longest = max(len(tokenizer(build_target(args.format, args.lang, ex.labels))["input_ids"]) for ex in train)
            args.max_new_tokens = min(512, longest + 32)
        train_ds, dev_ds = ChatSFTDataset(train, tokenizer, args), ChatSFTDataset(dev, tokenizer, args)
        gpu = torch.cuda.is_available()
        bf16 = gpu and torch.cuda.is_bf16_supported()
        targs = training_arguments(args, len(train_ds), seed, output_dir, gpu, bf16)
        tracker = BestAdapterTracker()
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                          data_collator=PadCollator(tokenizer.pad_token_id),
                          callbacks=[tracker, EarlyStoppingCallback(early_stopping_patience=args.patience)])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_train = tracker.t0 = time.time()
        trainer.train()
        if args.max_steps:  # epoch-based evaluation never fires when training stops by step count
            tracker.on_evaluate(targs, trainer.state, None, metrics=trainer.evaluate(), model=model)
        train_s = time.time() - t_train
        tracker.restore(model)
        logger.info("[seed %d] best dev loss %.4f at epoch %s", seed, tracker.best_loss or float("nan"), tracker.best_epoch)

        t_gen = time.time()
        preds, raw = generate(model, tokenizer, test, args, categories)
        gen_s = time.time() - t_gen
        metrics = micro_prf([ex.labels for ex in test], preds)
        n_tokens = sum(len(x["input_ids"]) for x in train_ds.items)
        epochs_done = float(trainer.state.epoch or 0)
        info = {"seed": seed, "best_epoch": tracker.best_epoch, "best_dev_loss": tracker.best_loss,
                "epochs_completed": round(epochs_done, 2), "max_epochs": args.epochs, "stopped_early": epochs_done < args.epochs - 1e-6,
                "global_steps": trainer.state.global_step,
                "steps_per_epoch": math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)),
                "dev_history": tracker.history,
                "train_log": [jsonable(h) for h in trainer.state.log_history],
                "load_seconds": round(t_train - t0, 1), "train_seconds": round(train_s, 1), "generate_seconds": round(gen_s, 1),
                "generate_ms_per_review": round(1000 * gen_s / max(1, len(test)), 1), "total_seconds": round(time.time() - t0, 1),
                "train_tokens_per_epoch": n_tokens, "train_tokens_per_second": round(n_tokens * epochs_done / max(train_s, 1e-9), 1),
                "max_new_tokens": args.max_new_tokens, "empty_predictions": sum(1 for p in preds if not p),
                "test_metrics": metrics, "output_stats": output_stats(raw, test, args, categories),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None}
        if not run_config:  # identical for every seed apart from the seed itself
            run_config.update({
                "command": " ".join([sys.executable] + sys.argv), "code": code_version(), "hardware": hardware(),
                "packages": package_versions(), "model": model_record(model, args),
                "training_arguments": jsonable(targs.to_dict()), "script_arguments": jsonable(vars(args)),
                "early_stopping": {"metric": "dev loss (answer tokens)", "patience_epochs": args.patience,
                                   "max_epochs": args.epochs, "selection": "epoch with the lowest dev loss"},
                "decoding": {"strategy": "greedy", "do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                             "batch_size": args.eval_batch_size, "padding_side": "left"},
                "prompt": {"format": args.format, "lang": args.lang, "domain_phrase": args.domain,
                           "chat_prompt_template": chat_prompt(tokenizer, args, "{review}"),
                           "target_example": build_target(args.format, args.lang, train[0].labels) + tokenizer.eos_token,
                           "loss_on": "answer tokens only (prompt masked with -100)"},
                "data": {"train_path": args.train_path, "dev_path": args.dev_path, "test_path": args.test_path,
                         "sha256": {k: file_sha256(v) for k, v in (("train", args.train_path), ("dev", args.dev_path), ("test", args.test_path))},
                         "n_train": len(train), "n_dev": len(dev), "n_test": len(test), "limit": args.limit,
                         "n_categories": len(categories), "categories": categories,
                         "train_lengths": train_ds.length_stats(), "dev_lengths": dev_ds.length_stats(),
                         "max_seq_length": args.max_seq_length}})
        logger.info("[seed %d] test P/R/F1 %.2f / %.2f / %.2f  (train %.0f s, generate %.0f s, %d empty)", seed,
                    metrics["precision"], metrics["recall"], metrics["f1"], train_s, gen_s, info["empty_predictions"])
        per_seed.append({"seed": seed, "metrics": metrics, "predictions": preds, "raw": raw, "info": info})
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(per_seed, key=lambda r: r["metrics"]["f1"])
    write_predictions(output_dir / "test_predictions.jsonl", test, best["predictions"])
    with (output_dir / "test_generations.jsonl").open("w", encoding="utf-8") as f:  # raw model text, for re-parsing later
        for ex, t in zip(test, best["raw"]):
            f.write(json.dumps({"id": ex.sample_id, "generation": t}, ensure_ascii=False) + "\n")
    write_metrics(output_dir / "multi_seed_summary.json", {
        "best_seed": best["seed"], "best": best["metrics"], "per_seed": [{"seed": r["seed"], **r["metrics"]} for r in per_seed],
        "domain": args.domain, "format": args.format, "lang": args.lang, "model_name": args.model_name, "backend": args.backend,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "load_in_4bit": args.load_in_4bit, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size * args.grad_accum})
    write_metrics(output_dir / "history.json", [r["info"] for r in per_seed])
    write_metrics(output_dir / "run_config.json", run_config)
    logger.info("Wrote %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen2.5-3B-Instruct + LoRA (Unsloth) instruction-tuned ACSA baseline")
    p.add_argument("--domain", required=True, help="domain phrase used inside the prompt, e.g. 'hotel' or 'beauty product'")
    p.add_argument("--format", choices=["code", "nl"], required=True)
    p.add_argument("--lang", choices=["vi", "en"], required=True)
    p.add_argument("--train_path", required=True)
    p.add_argument("--dev_path", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seeds", default="42")
    p.add_argument("--backend", choices=["unsloth", "hf"], default=None, help="default: unsloth with CUDA, hf otherwise")
    p.add_argument("--model_name", default=None)
    p.add_argument("--tokenizer_name", default=None, help="hf backend only (tiny test models)")
    p.add_argument("--load_in_4bit", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=0, help="0: longest training answer + 32")
    p.add_argument("--epochs", type=int, default=3, help="maximum; early stopping on dev loss (--patience) keeps the best epoch")
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=32, help="generation batch size")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0, help="smoke test: use only this many training reviews")
    p.add_argument("--max_steps", type=int, default=0, help="smoke test: stop after this many optimizer steps")
    return p


if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    a.backend = a.backend or ("unsloth" if torch.cuda.is_available() else "hf")
    a.model_name = a.model_name or DEFAULT_MODEL[a.backend]
    if a.backend == "unsloth" and not a.load_in_4bit and a.model_name.endswith("-bnb-4bit"):
        a.model_name = a.model_name[:-len("-bnb-4bit")]  # 16-bit LoRA needs the unquantized checkpoint
    run(a)
