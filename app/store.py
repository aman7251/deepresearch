"""Tiny JSON-file run store. Good enough for an MVP; swap for Redis/SQLite later."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.models import RunResult, RunStatus

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def _path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def save(result: RunResult) -> None:
    result.updated_at = datetime.now(timezone.utc).isoformat()
    _path(result.run_id).write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load(run_id: str) -> RunResult | None:
    p = _path(run_id)
    if not p.exists():
        return None
    return RunResult.model_validate_json(p.read_text(encoding="utf-8"))


def create(run_id: str, question: str) -> RunResult:
    result = RunResult(run_id=run_id, question=question, status=RunStatus.queued)
    save(result)
    return result


def append_log(run_id: str, msg: str) -> None:
    result = load(run_id)
    if result is None:
        return
    result.log.append(msg)
    save(result)
