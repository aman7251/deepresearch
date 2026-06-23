"""Pydantic schemas shared across API, worker and UI."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class Source(BaseModel):
    n: int  # citation number, 1-indexed
    url: str
    title: str = ""
    snippet: str = ""


class Verdict(str, Enum):
    supported = "supported"
    unsupported = "unsupported"
    contradicted = "contradicted"


class Claim(BaseModel):
    text: str
    cited_sources: list[int] = Field(default_factory=list)
    verdict: Verdict = Verdict.unsupported
    confidence: float = 0.0  # 0..1
    rationale: str = ""


class ResearchRequest(BaseModel):
    question: str = Field(min_length=4)


class RunResult(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.queued
    question: str = ""
    plan: list[str] = Field(default_factory=list)
    report_md: str = ""
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)
    error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def faithfulness(self) -> float:
        """Share of claims that are supported by their cited sources."""
        if not self.claims:
            return 0.0
        good = sum(1 for c in self.claims if c.verdict == Verdict.supported)
        return round(good / len(self.claims), 3)

    @property
    def citation_coverage(self) -> float:
        """Share of claims that carry at least one citation."""
        if not self.claims:
            return 0.0
        cited = sum(1 for c in self.claims if c.cited_sources)
        return round(cited / len(self.claims), 3)
