#!/usr/bin/env python3
"""Training-cost and inference-latency measurement shared by every training script (``--profile_json``).

Each script's profile mode trains its own model for ONE epoch with its normal configuration and code path, then
times inference on test reviews, and writes one JSON record. Definitions are identical across scripts:

* ``epoch_seconds``  wall-clock of the training loop of one epoch (no dev/test evaluation), CUDA-synchronised.
* ``peak_gpu_gb``    maximum memory allocated by tensors during that epoch (weights + optimizer state + activations),
                     summed over every GPU the model uses (large T5 models are split across two GPUs).
* ``latency_ms``     end-to-end per review at batch size 1: preprocessing (word segmentation / tokenization), the
                     forward pass or generation, and decoding into (category, polarity) pairs; first ``warmup``
                     reviews excluded.
* ``throughput``     reviews per second when the same reviews are processed at the script's evaluation batch size.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch


def count_params(*models) -> Dict[str, int]:
    total = sum(p.numel() for m in models for p in m.parameters())
    trainable = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def _devices() -> List[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def sync() -> None:
    for d in _devices():
        torch.cuda.synchronize(d)


def reset_peak() -> None:
    sync()
    for d in _devices():
        torch.cuda.reset_peak_memory_stats(d)


def peak_gpu() -> Dict[str, object]:
    sync()
    per = [torch.cuda.max_memory_allocated(d) / 1e9 for d in _devices()]
    used = [round(x, 3) for x in per if x > 0.01]  # devices the model actually touched
    return {"peak_gpu_gb": round(sum(used), 3), "peak_gpu_gb_per_device": used}


def now() -> float:
    sync()
    return time.perf_counter()


def latency(predict_one: Callable[[object], object], items: Sequence[object], warmup: int = 10) -> Dict[str, float]:
    """Per-item wall-clock (ms) of ``predict_one`` at batch size 1; the first ``warmup`` items are excluded."""
    times = []
    with torch.no_grad():
        for i, item in enumerate(items):
            t0 = now()
            predict_one(item)
            dt = (now() - t0) * 1000.0
            if i >= warmup:
                times.append(dt)
    times.sort()
    return {"latency_ms_mean": round(statistics.mean(times), 2), "latency_ms_median": round(statistics.median(times), 2),
            "latency_ms_p90": round(times[int(0.9 * (len(times) - 1))], 2), "latency_n": len(times)}


def throughput(predict_batch: Callable[[Sequence[object]], object], items: Sequence[object], batch_size: int) -> Dict[str, float]:
    """Reviews per second over ``items`` processed ``batch_size`` at a time (one warm-up batch first)."""
    with torch.no_grad():
        predict_batch(items[:batch_size])
        t0 = now()
        for i in range(0, len(items), batch_size):
            predict_batch(items[i:i + batch_size])
        dt = now() - t0
    return {"throughput_reviews_per_s": round(len(items) / dt, 2), "throughput_batch_size": batch_size}


def write(path: str, record: Dict) -> None:
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "gpus": [torch.cuda.get_device_name(d) for d in _devices()]}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    out = {**record, "env": env}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[profile] wrote {path}: " + ", ".join(f"{k}={v}" for k, v in out.items() if k not in ("env",)))
