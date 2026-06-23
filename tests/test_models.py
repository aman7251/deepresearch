"""Tests for RunResult derived metrics."""
from __future__ import annotations

from app.models import Claim, RunResult, Verdict


def _claim(text: str, verdict: Verdict, cites: list[int]) -> Claim:
    return Claim(text=text, verdict=verdict, cited_sources=cites, confidence=0.8)


def test_faithfulness_and_coverage():
    r = RunResult(
        run_id="t1",
        claims=[
            _claim("a", Verdict.supported, [1]),
            _claim("b", Verdict.supported, [2]),
            _claim("c", Verdict.unsupported, []),
            _claim("d", Verdict.contradicted, [3]),
        ],
    )
    assert r.faithfulness == 0.5  # 2 of 4 supported
    assert r.citation_coverage == 0.75  # 3 of 4 carry a citation


def test_empty_claims_metrics_are_zero():
    r = RunResult(run_id="t2", claims=[])
    assert r.faithfulness == 0.0
    assert r.citation_coverage == 0.0


def test_demo_sample_round_trips():
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "samples" / "demo_run.json"
    r = RunResult.model_validate_json(p.read_text(encoding="utf-8"))
    assert r.status.value == "done"
    assert r.claims
    # the demo intentionally contains one flagged (contradicted) claim
    assert any(c.verdict != Verdict.supported for c in r.claims)
