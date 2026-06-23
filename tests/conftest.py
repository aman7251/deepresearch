"""Shared fixtures and fakes so the whole agent loop is testable offline.

No network, no API keys, no model downloads: the LLM, web search, page reader and
vector index are all replaced with deterministic fakes.
"""
from __future__ import annotations

from typing import Any, Callable

import pytest

from app.llm import UsageTracker


class FakeLLM:
    """Drop-in replacement for app.llm.LLMClient.

    `responses` maps a node name -> either a static value or a callable
    (messages, calls_so_far) -> value.
    """

    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
        self.tracker = UsageTracker()
        self.model = "fake-model"
        self.calls: list[str] = []

    def _record(self, node: str) -> None:
        self.calls.append(node)
        # simulate token usage so the usage/cost dashboard has data
        self.tracker.add(node, prompt=100, completion=20, model="fake-model", latency=0.01)

    def chat_json(self, messages: list[dict], node: str = "llm", temperature: float = 0.1) -> Any:
        self._record(node)
        r = self.responses[node]
        return r(messages, self.calls) if callable(r) else r

    def chat(self, messages: list[dict], node: str = "llm", temperature: float = 0.2, json_mode: bool = False) -> str:
        self._record(node)
        return "ok"


class FakeIndex:
    """In-memory stand-in for app.rag.EphemeralIndex."""

    def __init__(self) -> None:
        self.docs: list[tuple[int, str, str]] = []

    def add_document(self, text: str, source_n: int, url: str) -> int:
        if not text:
            return 0
        self.docs.append((source_n, url, text))
        return 1

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        return [
            {"text": t[:200], "source_n": n, "url": u, "score": 0.9}
            for (n, u, t) in self.docs[:k]
        ]

    @property
    def size(self) -> int:
        return len(self.docs)


@pytest.fixture
def patch_tools(monkeypatch) -> None:
    """Replace web search + page reader with deterministic fakes."""
    import app.agents as agents

    def fake_search(query: str, max_results: int = 4) -> list[dict]:
        return [
            {"title": "Source A", "url": "https://a.example/x", "snippet": "snippet a"},
            {"title": "Source B", "url": "https://b.example/y", "snippet": "snippet b"},
        ]

    def fake_fetch(url: str, **kwargs) -> str:
        return f"Readable content for {url}. HTTP/3 uses QUIC over UDP. " * 20

    monkeypatch.setattr(agents, "web_search", fake_search)
    monkeypatch.setattr(agents, "fetch_and_extract", fake_fetch)


@pytest.fixture
def make_llm() -> Callable[..., FakeLLM]:
    def _build(critic_first_pass: str = "supported") -> FakeLLM:
        def planner(_messages, _calls):
            return {"subquestions": ["What is QUIC?", "How does HTTP/2 multiplex?"]}

        def synthesizer(_messages, _calls):
            return {
                "report_md": "HTTP/3 runs over QUIC/UDP [1]. HTTP/2 multiplexes over TCP [2].",
                "claims": [
                    {"text": "HTTP/3 runs over QUIC/UDP.", "cited_sources": [1]},
                    {"text": "HTTP/2 multiplexes over TCP.", "cited_sources": [2]},
                ],
            }

        def critic(_messages, calls):
            # On the first synthesis pass, optionally flag a claim; after revision, all pass.
            synth_count = calls.count("synthesizer")
            if synth_count <= 1 and critic_first_pass != "supported":
                return {"verdict": critic_first_pass, "confidence": 0.3, "rationale": "weak evidence"}
            return {"verdict": "supported", "confidence": 0.92, "rationale": "evidence matches"}

        return FakeLLM({"planner": planner, "synthesizer": synthesizer, "critic": critic})

    return _build
