#!/usr/bin/env python3
"""Summary of the Qwen2.5-3B instruction-tuning runs (baselines/qwen_instruction_tuning.py) for the paper.

Reads outputs/qwen_<domain>_<format>_<lang>/seed_<s>/{multi_seed_summary,run_config,history}.json and prints:
  1. test micro-F1 (mean +- std over seeds) per variant and domain, plus the five-domain average (Table 4 rows);
  2. per variant and domain: selected epochs, early stops, training/generation time, peak memory, output-format errors;
  3. whether every run used the same model revision, hyperparameters and package versions (lists any differences);
  4. the hyperparameters to report (Table 2) and the hardware.
Writes the same content to outputs/qwen_report.json.

    python3 scripts/qwen_run_report.py [--root outputs] [--seeds 42,123,2024,7,99]
"""
import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = ["restaurant", "hotel", "phone", "education", "beauty"]
VARIANTS = [("code", "en"), ("code", "vi"), ("nl", "en"), ("nl", "vi")]
NAMES = {("code", "en"): "Code Instruction-En", ("code", "vi"): "Code Instruction-Vi",
         ("nl", "en"): "NL Instruction-En", ("nl", "vi"): "NL Instruction-Vi"}
# Settings that must be identical across runs (seed and run-specific paths excluded).
SHARED = {
    "model": ["model_name", "model_revision", "load_in_4bit", "quantization_config", "lora_config", "trainable_parameters",
              "total_parameters", "gradient_checkpointing"],
    # warmup_steps and max_new_tokens are derived per domain (10% of the steps; longest answer + 32), so the script
    # arguments they come from are compared instead.
    "script_arguments": ["model_name", "load_in_4bit", "lora_r", "lora_alpha", "learning_rate", "weight_decay", "warmup_ratio",
                         "epochs", "patience", "batch_size", "grad_accum", "max_seq_length", "eval_batch_size", "limit", "max_steps"],
    "training_arguments": ["learning_rate", "lr_scheduler_type", "num_train_epochs", "per_device_train_batch_size",
                           "gradient_accumulation_steps", "optim", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon",
                           "max_grad_norm", "fp16", "bf16", "group_by_length", "train_sampling_strategy"],
    "early_stopping": None, "decoding": ["strategy", "num_beams", "do_sample", "batch_size"],
    "packages": None, "hardware": ["device", "gpu_memory_gb"],
}


def load_runs(root: Path, seeds):
    runs = []
    for (fmt, lang) in VARIANTS:
        for d in DOMAINS:
            for s in seeds:
                rd = root / f"qwen_{d}_{fmt}_{lang}" / f"seed_{s}"
                if not (rd / "multi_seed_summary.json").exists():
                    continue
                summ = json.loads((rd / "multi_seed_summary.json").read_text())
                cfg = json.loads((rd / "run_config.json").read_text()) if (rd / "run_config.json").exists() else {}
                hist = json.loads((rd / "history.json").read_text()) if (rd / "history.json").exists() else [{}]
                runs.append({"variant": (fmt, lang), "domain": d, "seed": s, "f1": summ["best"]["f1"],
                             "precision": summ["best"]["precision"], "recall": summ["best"]["recall"],
                             "cfg": cfg, "hist": hist[0] if hist else {}, "dir": str(rd)})
    return runs


def ms(vals):
    if not vals:
        return "--"
    return f"{st.mean(vals):.2f}" + (f"+-{st.stdev(vals):.2f}" if len(vals) > 1 else "")


def pick(d, keys):
    return {k: d.get(k) for k in keys} if keys else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--seeds", default="42,123,2024,7,99")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = load_runs(Path(a.root), seeds)
    if not runs:
        print(f"no Qwen runs under {a.root}/qwen_*")
        return
    by = defaultdict(list)
    for r in runs:
        by[(r["variant"], r["domain"])].append(r)
    report = {"runs": len(runs), "results": {}, "per_cell": {}, "consistency": {}, "hyperparameters": {}}

    print(f"== 1. Test micro-F1, mean+-std over seeds ({len(runs)} runs; n = seeds found)")
    print(f"{'variant':22s}" + "".join(f"{d:>21s}" for d in DOMAINS) + f"{'Average':>10s}")
    for v in VARIANTS:
        cells, means = [], []
        for d in DOMAINS:
            f = [r["f1"] for r in by.get((v, d), [])]
            cells.append(f"{ms(f)} (n={len(f)})" if f else "--")
            if f:
                means.append(st.mean(f))
            report["results"].setdefault(NAMES[v], {})[d] = {"mean": st.mean(f) if f else None,
                                                            "std": st.stdev(f) if len(f) > 1 else None, "n": len(f), "f1": f}
        avg = f"{st.mean(means):.2f}" if len(means) == len(DOMAINS) else "--"
        report["results"][NAMES[v]]["average"] = st.mean(means) if len(means) == len(DOMAINS) else None
        print(f"{NAMES[v]:22s}" + "".join(f"{c:>21s}" for c in cells) + f"{avg:>10s}")

    print("\n== 2. Per variant and domain: selected epoch, early stops, time, memory, output format")
    print(f"{'variant':22s}{'domain':11s}{'n':>3s} {'best epochs':14s}{'early':>6s}{'train min':>10s}{'gen min':>8s}"
          f"{'peak GB':>8s}{'tok/s':>7s}{'empty':>6s}{'unk.cat':>8s}{'bad sent':>9s}")
    for v in VARIANTS:
        for d in DOMAINS:
            rs = by.get((v, d), [])
            if not rs:
                continue
            h = [r["hist"] for r in rs]
            eps = Counter(x.get("best_epoch") for x in h)
            row = {"n": len(rs), "best_epochs": dict(eps), "early_stops": sum(bool(x.get("stopped_early")) for x in h),
                   "train_min_mean": st.mean(x.get("train_seconds", 0) for x in h) / 60,
                   "generate_min_mean": st.mean(x.get("generate_seconds", 0) for x in h) / 60,
                   "peak_gpu_gb_max": max((x.get("peak_gpu_gb") or 0) for x in h),
                   "train_tokens_per_second_mean": st.mean(x.get("train_tokens_per_second", 0) for x in h),
                   "empty_predictions": sum(x.get("empty_predictions", 0) for x in h),
                   "pairs_unknown_category": sum(x.get("output_stats", {}).get("pairs_unknown_category", 0) for x in h),
                   "pairs_invalid_sentiment": sum(x.get("output_stats", {}).get("pairs_invalid_sentiment", 0) for x in h)}
            report["per_cell"][f"{NAMES[v]}/{d}"] = row
            print(f"{NAMES[v]:22s}{d:11s}{row['n']:3d} {str(dict(sorted(eps.items(), key=str))):14s}{row['early_stops']:6d}"
                  f"{row['train_min_mean']:10.1f}{row['generate_min_mean']:8.1f}{row['peak_gpu_gb_max']:8.2f}"
                  f"{row['train_tokens_per_second_mean']:7.0f}{row['empty_predictions']:6d}{row['pairs_unknown_category']:8d}"
                  f"{row['pairs_invalid_sentiment']:9d}")
    total_h = sum(r["hist"].get("total_seconds", 0) for r in runs) / 3600
    print(f"total GPU time of these runs: {total_h:.1f} h")
    report["total_gpu_hours"] = total_h

    print("\n== 3. Consistency across runs (every value below should be identical)")
    ok = True
    for section, keys in SHARED.items():
        seen = defaultdict(set)
        for r in runs:
            for k, val in pick(r["cfg"].get(section, {}), keys).items():
                if section == "lora_config" or k in ("lora_config", "quantization_config"):
                    val = {kk: vv for kk, vv in (val or {}).items() if kk not in ("base_model_name_or_path", "revision")}
                seen[k].add(json.dumps(val, sort_keys=True))
        for k, vals in seen.items():
            report["consistency"][f"{section}.{k}"] = sorted(vals)
            if len(vals) > 1:
                ok = False
                print(f"  DIFFERS {section}.{k}: {sorted(vals)}")
    commits = Counter(r["cfg"].get("code", {}).get("git_commit") for r in runs)
    dirty = sum(bool(r["cfg"].get("code", {}).get("git_dirty")) for r in runs)
    print(f"  code commits: {dict(commits)}; runs from a modified checkout: {dirty}")
    print("  all shared settings identical" if ok else "  -> differences above must be explained or the runs repeated")

    print("\n== 4. Hyperparameters to report (from the first run)")
    c = runs[0]["cfg"]
    m, t, e, dec, pr = c.get("model", {}), c.get("training_arguments", {}), c.get("early_stopping", {}), c.get("decoding", {}), c.get("prompt", {})
    sa = c.get("script_arguments", {})
    lora = m.get("lora_config", {})
    quant = m.get("quantization_config", {})
    hp = {
        "backbone": m.get("model_name"), "model_revision": m.get("model_revision"),
        "quantization": (f"4-bit {quant.get('bnb_4bit_quant_type', '')} (double quant {quant.get('bnb_4bit_use_double_quant')}), "
                         f"compute {quant.get('bnb_4bit_compute_dtype')}") if quant.get("load_in_4bit") else f"none ({m.get('dtype')})",
        "lora": {"r": lora.get("r"), "alpha": lora.get("lora_alpha"), "dropout": lora.get("lora_dropout"),
                 "target_modules": lora.get("target_modules"), "bias": lora.get("bias")},
        "trainable_parameters": m.get("trainable_parameters"), "total_parameters": m.get("total_parameters"),
        "optimizer": t.get("optim"), "learning_rate": t.get("learning_rate"), "weight_decay": t.get("weight_decay"),
        "adam_betas_eps": [t.get("adam_beta1"), t.get("adam_beta2"), t.get("adam_epsilon")], "max_grad_norm": t.get("max_grad_norm"),
        "lr_schedule": t.get("lr_scheduler_type"), "warmup": f"{sa.get('warmup_ratio', 0):.0%} of the training steps",
        "batch_size": (t.get("per_device_train_batch_size") or 0) * (t.get("gradient_accumulation_steps") or 1),
        "max_epochs": e.get("max_epochs"), "early_stopping": f"patience {e.get('patience_epochs')} on {e.get('metric')}",
        "precision": "fp16" if t.get("fp16") else ("bf16" if t.get("bf16") else "fp32"),
        "gradient_checkpointing": m.get("gradient_checkpointing"),
        "max_seq_length": c.get("data", {}).get("max_seq_length"), "decoding": dec,
        "loss": pr.get("loss_on"), "hardware": c.get("hardware"), "packages": c.get("packages"),
    }
    report["hyperparameters"] = hp
    for k, v in hp.items():
        print(f"  {k}: {v}")
    out = Path(a.root) / "qwen_report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
