"""LLM provider abstraction with built-in token + cost tracking.

Both Groq and Ollama expose an OpenAI-compatible API, so a single client works
for both. Switching providers is a one-line .env change (LLM_PROVIDER).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

# Approx Groq free/dev-tier prices (USD per 1M tokens). Local Ollama = $0.
# Used only to surface an order-of-magnitude cost in the UI, not for billing.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
}


@dataclass
class UsageTracker:
    """Accumulates token usage, cost and per-call latency across an agent run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    latency_s: float = 0.0
    by_node: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, node: str, prompt: int, completion: int, model: str, latency: float) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        self.latency_s += latency
        in_price, out_price = _PRICE_PER_MTOK.get(model, (0.0, 0.0))
        cost = (prompt / 1_000_000) * in_price + (completion / 1_000_000) * out_price
        self.cost_usd += cost
        n = self.by_node.setdefault(node, {"prompt": 0, "completion": 0, "cost_usd": 0.0, "calls": 0})
        n["prompt"] += prompt
        n["completion"] += completion
        n["cost_usd"] += cost
        n["calls"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "latency_s": round(self.latency_s, 2),
            "by_node": self.by_node,
        }


class LLMClient:
    """Thin wrapper over an OpenAI-compatible endpoint with usage tracking."""

    def __init__(self, settings: Settings | None = None, tracker: UsageTracker | None = None):
        self.settings = settings or get_settings()
        self.tracker = tracker or UsageTracker()
        self.model = self.settings.active_model
        self._client = OpenAI(
            api_key=self.settings.active_api_key or "missing-key",
            base_url=self.settings.active_base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
    def chat(
        self,
        messages: list[dict[str, str]],
        node: str = "llm",
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        import time

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start

        usage = resp.usage
        self.tracker.add(
            node=node,
            prompt=getattr(usage, "prompt_tokens", 0) or 0,
            completion=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
            latency=latency,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict[str, str]], node: str = "llm", temperature: float = 0.1) -> Any:
        """Chat call that must return JSON. Falls back to brace-extraction if the
        provider lacks strict JSON mode."""
        raw = self.chat(messages, node=node, temperature=temperature, json_mode=True)
        return _safe_json(raw)


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise
