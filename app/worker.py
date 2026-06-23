"""arq worker: runs the multi-minute agent loop off the HTTP request path.

Golden rule: never run a long agent loop inside an HTTP handler. The API enqueues
a job here; the UI polls the run store for status + result.
"""
from __future__ import annotations

from arq import cron  # noqa: F401  (kept for future scheduled evals)
from arq.connections import RedisSettings

from app.agents import run_research
from app.config import get_settings
from app.models import RunStatus
from app.store import load, save


async def research_job(ctx: dict, run_id: str, question: str) -> str:
    settings = get_settings()
    result = load(run_id)
    if result is None:
        return run_id
    result.status = RunStatus.running
    save(result)

    def emit(msg: str) -> None:
        current = load(run_id)
        if current is not None:
            current.log.append(msg)
            save(current)

    try:
        # run_research is synchronous (LangGraph .invoke); run in a thread so we
        # don't block the event loop.
        import anyio

        final = await anyio.to_thread.run_sync(
            lambda: run_research(question, run_id=run_id, settings=settings, emit=emit)
        )
        save(final)
    except Exception as exc:  # noqa: BLE001
        result = load(run_id) or result
        result.status = RunStatus.error
        result.error = f"{type(exc).__name__}: {exc}"
        save(result)
    return run_id


class WorkerSettings:
    functions = [research_job]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2
    job_timeout = 600
