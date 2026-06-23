"""Streamlit front-end for DeepResearch.

Run:  streamlit run app/ui_streamlit.py
Shows the cited report, a fact-check panel (flagged claims in red), and a
token/cost/latency observability panel. With DEMO_MODE=1 it loads a cached run
so the public demo works without any API keys.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import streamlit as st

from app.config import get_settings
from app.models import RunResult

settings = get_settings()
DEMO_PATH = Path(__file__).resolve().parent.parent / "samples" / "demo_run.json"

VERDICT_BADGE = {
    "supported": ":green[supported]",
    "unsupported": ":orange[unsupported]",
    "contradicted": ":red[contradicted]",
}

st.set_page_config(page_title="DeepResearch", page_icon="🔎", layout="wide")


def load_demo() -> RunResult:
    return RunResult.model_validate_json(DEMO_PATH.read_text(encoding="utf-8"))


def submit_and_poll(question: str, placeholder) -> RunResult | None:
    base = settings.api_base_url
    try:
        resp = httpx.post(f"{base}/research", json={"question": question}, timeout=30)
        resp.raise_for_status()
        run_id = resp.json()["run_id"]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the API at {base}: {exc}\n\nStart the API + worker, or enable DEMO_MODE.")
        return None

    for _ in range(180):  # up to ~6 min
        try:
            r = httpx.get(f"{base}/runs/{run_id}", timeout=15)
            result = RunResult.model_validate(r.json())
        except Exception:  # noqa: BLE001
            time.sleep(2)
            continue
        with placeholder.container():
            st.caption(f"Status: **{result.status}** · run `{run_id}`")
            if result.log:
                st.code("\n".join(result.log[-12:]), language="text")
        if result.status in ("done", "error"):
            return result
        time.sleep(2)
    st.warning("Timed out waiting for the run.")
    return None


def render_result(result: RunResult) -> None:
    if result.status == "error":
        st.error(f"Run failed: {result.error}")
        return

    # --- top metrics ---
    u = result.usage or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faithfulness", f"{result.faithfulness:.0%}", help="Claims supported by their sources")
    m2.metric("Citation coverage", f"{result.citation_coverage:.0%}", help="Claims carrying a citation")
    m3.metric("Tokens", f"{u.get('total_tokens', 0):,}")
    m4.metric("Est. cost", f"${u.get('cost_usd', 0):.4f}", help=f"{u.get('latency_s', 0)}s · {u.get('calls', 0)} LLM calls")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Report")
        st.markdown(result.report_md or "_no report_")

        st.subheader("Sources")
        for s in result.sources:
            st.markdown(f"**[{s.n}]** [{s.title or s.url}]({s.url})")

    with right:
        flagged = [c for c in result.claims if c.verdict.value != "supported"]
        st.subheader(f"Fact-check ({len(result.claims)} claims)")
        if flagged:
            st.error(f"{len(flagged)} claim(s) flagged as not fully supported.")
        else:
            st.success("All claims supported by sources.")

        for c in result.claims:
            badge = VERDICT_BADGE.get(c.verdict.value, c.verdict.value)
            cites = "".join(f"[{n}]" for n in c.cited_sources) or "(no citation)"
            with st.expander(f"{badge} · conf {c.confidence:.0%} · {c.text[:70]}"):
                st.write(c.text)
                st.caption(f"Cited sources: {cites}")
                if c.rationale:
                    st.caption(f"Verifier: {c.rationale}")

        if u.get("by_node"):
            st.subheader("Cost by agent")
            st.json(u["by_node"], expanded=False)


def main() -> None:
    st.title("🔎 DeepResearch")
    st.caption(
        "Ask a hard question. A planner -> search -> reader -> synthesizer -> critic loop "
        "writes a cited report and fact-checks every claim against its sources."
    )

    with st.sidebar:
        st.header("Settings")
        st.write(f"Provider: **{settings.llm_provider}**")
        st.write(f"Model: `{settings.active_model}`")
        st.write(f"Demo mode: **{'on' if settings.demo_mode else 'off'}**")
        if settings.demo_mode:
            st.info("Demo mode serves a cached sample run. Set DEMO_MODE=0 for live research.")

    default_q = "Compare the 2025 EU and US approaches to AI regulation."
    question = st.text_input("Research question", value=default_q if settings.demo_mode else "")
    go = st.button("Research", type="primary")

    placeholder = st.empty()

    if go and question.strip():
        if settings.demo_mode:
            with st.spinner("Loading cached demo run..."):
                time.sleep(0.6)
            render_result(load_demo())
        else:
            result = submit_and_poll(question.strip(), placeholder)
            if result:
                render_result(result)
    elif settings.demo_mode:
        st.divider()
        st.caption("Preview of a cached run:")
        render_result(load_demo())


if __name__ == "__main__":
    main()
