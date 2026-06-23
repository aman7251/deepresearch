"""End-to-end tests of the LangGraph loop with all I/O faked (offline, no keys)."""
from __future__ import annotations

from app.agents import run_research
from app.models import Verdict
from tests.conftest import FakeIndex


def test_happy_path_runs_full_loop(patch_tools, make_llm):
    llm = make_llm(critic_first_pass="supported")
    result = run_research("Differences between HTTP/2 and HTTP/3?", run_id="t-happy", llm=llm, index=FakeIndex())

    assert result.status.value == "done"
    assert result.plan == ["What is QUIC?", "How does HTTP/2 multiplex?"]
    assert result.sources  # search produced sources
    assert len(result.claims) == 2
    assert all(c.verdict == Verdict.supported for c in result.claims)
    assert result.faithfulness == 1.0
    assert result.citation_coverage == 1.0

    # observability populated
    assert result.usage["total_tokens"] > 0
    assert "planner" in result.usage["by_node"]

    # nodes executed in order; no revision needed
    assert llm.calls.count("synthesizer") == 1
    assert "planner" in llm.calls and "critic" in llm.calls


def test_unsupported_claim_triggers_one_revision(patch_tools, make_llm):
    llm = make_llm(critic_first_pass="unsupported")
    result = run_research("test", run_id="t-revise", llm=llm, index=FakeIndex())

    # the critic flagged on the first pass -> synthesizer should run twice (revise once)
    assert llm.calls.count("synthesizer") == 2
    # after revision the verifier passes everything
    assert all(c.verdict == Verdict.supported for c in result.claims)
    assert result.status.value == "done"


def test_log_traces_each_agent(patch_tools, make_llm):
    llm = make_llm()
    result = run_research("test", run_id="t-log", llm=llm, index=FakeIndex())
    joined = "\n".join(result.log)
    assert "Planner" in joined
    assert "Search" in joined
    assert "Reader" in joined
    assert "Synthesizer" in joined
    assert "Critic/Verifier" in joined
