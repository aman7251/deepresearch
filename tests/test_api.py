"""Smoke test for the FastAPI surface (no Redis needed).

Runs in demo mode so the lifespan never tries to open a Redis pool.
"""
from __future__ import annotations

import importlib


def _client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DEMO_MODE", "1")
    from app import config

    config.get_settings.cache_clear()
    import app.api as api

    importlib.reload(api)
    return TestClient(api.app)


def test_health_ok(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert body["queue"] is False  # no Redis in tests


def test_unknown_run_404(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.get("/runs/does-not-exist")
        assert resp.status_code == 404
