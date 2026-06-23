"""Unit tests for the LLM usage tracker and JSON parsing helpers."""
from __future__ import annotations

import pytest

from app.llm import UsageTracker, _safe_json


def test_usage_tracker_accumulates_and_prices():
    t = UsageTracker()
    t.add("planner", prompt=1_000_000, completion=1_000_000, model="llama-3.3-70b-versatile", latency=1.0)
    d = t.as_dict()
    assert d["prompt_tokens"] == 1_000_000
    assert d["completion_tokens"] == 1_000_000
    assert d["total_tokens"] == 2_000_000
    # 0.59 in + 0.79 out per Mtok
    assert d["cost_usd"] == pytest.approx(1.38, rel=1e-3)
    assert d["calls"] == 1
    assert "planner" in d["by_node"]


def test_usage_tracker_unknown_model_is_free():
    t = UsageTracker()
    t.add("critic", prompt=500, completion=500, model="ollama-local", latency=0.5)
    assert t.as_dict()["cost_usd"] == 0.0


def test_safe_json_plain():
    assert _safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_embedded_object():
    raw = 'Sure! Here is the result:\n{"verdict": "supported", "confidence": 0.9}\nThanks.'
    assert _safe_json(raw)["verdict"] == "supported"


def test_safe_json_embedded_array():
    raw = "noise [1, 2, 3] trailing"
    assert _safe_json(raw) == [1, 2, 3]
