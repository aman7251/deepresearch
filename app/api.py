"""FastAPI surface: enqueue research jobs and serve their status/results."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.models import ResearchRequest, RunResult
from app.store import create, load

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = None
    if not settings.demo_mode:
        try:
            app.state.redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        except Exception:  # noqa: BLE001 — allow API to boot even if Redis is down
            app.state.redis = None
    yield
    if app.state.redis is not None:
        await app.state.redis.close()


app = FastAPI(title="DeepResearch API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "demo_mode": settings.demo_mode,
        "provider": settings.llm_provider,
        "model": settings.active_model,
        "queue": app.state.redis is not None,
    }


@app.post("/research", response_model=RunResult)
async def research(req: ResearchRequest) -> RunResult:
    run_id = uuid.uuid4().hex[:12]
    result = create(run_id, req.question)
    if app.state.redis is None:
        raise HTTPException(
            status_code=503,
            detail="Job queue unavailable. Start Redis + the arq worker, or use DEMO_MODE.",
        )
    await app.state.redis.enqueue_job("research_job", run_id, req.question)
    return result


@app.get("/runs/{run_id}", response_model=RunResult)
async def get_run(run_id: str) -> RunResult:
    result = load(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result
