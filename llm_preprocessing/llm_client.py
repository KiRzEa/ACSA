#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM wrapper shared by all 3 LLM-preprocessing datasets, supporting 2
providers behind the same call_json() interface: OpenAI (proven, working)
and AWS Bedrock (added as an alternative -- Bedrock account/model-access
issues are a separate, account-level problem, not something this code can
route around). Loads secrets from .env (OPENAI_API_KEY, and/or
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN -- never printed/
logged), retries on transient errors, enforces JSON output, and tracks real
token usage/cost for every call (persisted to a log file, not just held in
memory -- a long run getting interrupted shouldn't lose the cost record).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

# USD per 1M tokens, standard (non-batch) pricing, verified via web search
# 2026-08-22. Re-check before trusting for other models / after a long gap --
# both OpenAI and AWS have changed prices before.
PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5-nano": {"input": 0.050, "output": 0.400},
    "amazon.nova-micro-v1:0": {"input": 0.035, "output": 0.140},
}


def load_dotenv(path: str | Path = None) -> None:
    """Minimal .env loader (no python-dotenv dependency -- may not be
    available on Kaggle). Only sets keys not already in the environment.
    Populates whatever keys are present -- OPENAI_API_KEY and/or the AWS_*
    triple -- boto3's default credential chain picks up AWS_ACCESS_KEY_ID/
    AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN from os.environ natively, no
    extra plumbing needed here."""
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()
_openai_client = None
_bedrock_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (checked .env and environment)")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_bedrock_client(region: str = "us-east-1"):
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return _bedrock_client


class CostTracker:
    """Accumulates token usage/cost across calls and persists after every
    single call (append-only JSON), so an interrupted run still has an
    accurate record of what it actually spent. Thread-safe: record() is
    called from worker threads under concurrency (apply phase), and the
    read-modify-write on the log file would race/corrupt without a lock."""

    def __init__(self, log_path: str | Path = "outputs_llm/cost_log.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "by_model": {}}
        if self.log_path.exists():
            self.totals = json.loads(self.log_path.read_text(encoding="utf-8"))
        self._lock = threading.Lock()

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = PRICING_PER_1M.get(model)
        cost = 0.0
        if price:
            cost = prompt_tokens / 1_000_000 * price["input"] + completion_tokens / 1_000_000 * price["output"]
        else:
            print(f"WARNING: no pricing entry for model={model!r}, cost not tracked for this call")

        with self._lock:
            self.totals["calls"] += 1
            self.totals["prompt_tokens"] += prompt_tokens
            self.totals["completion_tokens"] += completion_tokens
            self.totals["cost_usd"] += cost
            by_model = self.totals["by_model"].setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            by_model["calls"] += 1
            by_model["prompt_tokens"] += prompt_tokens
            by_model["completion_tokens"] += completion_tokens
            by_model["cost_usd"] += cost
            self.log_path.write_text(json.dumps(self.totals, ensure_ascii=False, indent=2), encoding="utf-8")
        return cost

    def summary(self) -> str:
        t = self.totals
        return f"{t['calls']} calls, {t['prompt_tokens']+t['completion_tokens']:,} tokens, ${t['cost_usd']:.4f} total"


def combined_cost(*log_paths: str | Path) -> dict:
    """Sum several scoped cost logs together -- e.g. develop + apply cost
    for one domain's Dataset 1 = its true total cost end to end."""
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    for p in log_paths:
        p = Path(p)
        if not p.exists():
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        for k in total:
            total[k] += t.get(k, 0)
    return total


_trackers: dict = {}


def get_tracker(log_path: str | Path) -> CostTracker:
    """Every call site must say explicitly which scoped cost log it belongs
    to (e.g. outputs_llm/hotel_dataset1/cost_develop.json) -- no hidden
    global default, so cost is never silently mixed across domains/phases."""
    key = str(Path(log_path).resolve())
    if key not in _trackers:
        _trackers[key] = CostTracker(log_path)
    return _trackers[key]


def _extract_json(content: str) -> dict:
    """Bedrock's Converse API has no OpenAI-style hard JSON-mode guarantee
    across every model family, so the prompt asks for JSON but the model may
    still wrap it in a markdown fence or add a stray preamble. Strip fences,
    then fall back to grabbing the first balanced-looking {...} block."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


_no_temperature_models: set = set()
_no_temperature_lock = threading.Lock()


def _call_openai(system_prompt: str, user_prompt: str, model: str, temperature: float) -> tuple[dict, int, int]:
    """Some models (e.g. gpt-5-nano) reject any temperature other than their
    default (1) with a 400 error -- detected once per model and cached in
    _no_temperature_models so later calls skip straight to omitting it."""
    client = get_openai_client()
    kwargs = {}
    if model not in _no_temperature_models:
        kwargs["temperature"] = temperature
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
    except Exception as e:
        if "temperature" in str(e) and "unsupported_value" in str(e) and model not in _no_temperature_models:
            with _no_temperature_lock:
                _no_temperature_models.add(model)
            print(f"NOTE: model={model!r} doesn't support custom temperature, retrying without it")
            return _call_openai(system_prompt, user_prompt, model, temperature)
        raise
    usage = response.usage
    content = response.choices[0].message.content
    return json.loads(content), usage.prompt_tokens, usage.completion_tokens


def _call_bedrock(system_prompt: str, user_prompt: str, model: str, temperature: float, region: str) -> tuple[dict, int, int]:
    client = get_bedrock_client(region)
    # No native hard-JSON-mode guarantee across every Bedrock model family
    # (unlike OpenAI's response_format) -- ask for JSON explicitly in-prompt.
    system_prompt = system_prompt + "\n\nTrả lời CHỈ bằng JSON hợp lệ, không kèm markdown code fence, không thêm chữ nào khác ngoài JSON."
    response = client.converse(
        modelId=model,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"temperature": temperature},
    )
    content = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    return _extract_json(content), usage["inputTokens"], usage["outputTokens"]


def call_json(
    system_prompt: str,
    user_prompt: str,
    tracker: CostTracker,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    temperature: float = 0.0,
    max_retries: int = 5,
    aws_region: str = "us-east-1",
) -> dict:
    """Call the model, forcing a JSON object response. Retries with
    exponential backoff on transient errors (rate limits, timeouts, 5xx).
    Every successful call's token usage/cost is recorded into `tracker` --
    always pass the scoped tracker for the phase/domain currently running,
    via get_tracker(log_path). provider is "openai" or "bedrock"."""
    last_error = None
    for attempt in range(max_retries):
        try:
            if provider == "bedrock":
                result, prompt_tokens, completion_tokens = _call_bedrock(system_prompt, user_prompt, model, temperature, aws_region)
            elif provider == "openai":
                result, prompt_tokens, completion_tokens = _call_openai(system_prompt, user_prompt, model, temperature)
            else:
                raise ValueError(f"Unknown provider {provider!r}, expected 'openai' or 'bedrock'")
            tracker.record(model, prompt_tokens, completion_tokens)
            return result
        except (json.JSONDecodeError,) as e:
            last_error = e  # malformed JSON -- retry, model may do better
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                break
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"call_json failed after {max_retries} attempts: {last_error}")
