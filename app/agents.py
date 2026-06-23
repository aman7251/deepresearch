"""The multi-agent research loop, built as a LangGraph state machine.

  planner -> search -> reader -> synthesizer -> critic/verifier
                                      ^                |
                                      |  (revise once) |
                                      +----------------+

The differentiator is the critic/verifier node: it fact-checks every claim in the
draft against the ephemeral RAG index (the actual fetched sources) and labels each
claim supported / unsupported / contradicted with a confidence score. If claims are
unsupported, it sends the draft back to the synthesizer once for revision.
"""
from __future__ import annotations

import json
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings, get_settings
from app.llm import LLMClient
from app.models import Claim, RunResult, Source, Verdict
from app.rag import EphemeralIndex
from app.tools import fetch_and_extract, web_search

MAX_REVISIONS = 1


class AgentState(TypedDict, total=False):
    question: str
    plan: list[str]
    sources: list[dict]
    report_md: str
    claims: list[dict]
    revision: int
    critic_feedback: str
    log: list[str]
    # runtime-only objects (not serialized)
    _llm: LLMClient
    _index: EphemeralIndex
    _settings: Settings
    _emit: Callable[[str], None]


def _log(state: AgentState, msg: str) -> None:
    state.setdefault("log", []).append(msg)
    emit = state.get("_emit")
    if emit:
        emit(msg)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def planner_node(state: AgentState) -> AgentState:
    llm, settings = state["_llm"], state["_settings"]
    _log(state, "Planner: decomposing the question into sub-questions...")
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a research planner. Break the user's question into a small set "
                "of focused, searchable sub-questions that together fully answer it. "
                f"Return JSON: {{\"subquestions\": [string, ...]}} with at most "
                f"{settings.max_subquestions} items."
            ),
        },
        {"role": "user", "content": state["question"]},
    ]
    data = llm.chat_json(prompt, node="planner")
    plan = [s.strip() for s in (data.get("subquestions") or []) if s.strip()][: settings.max_subquestions]
    if not plan:
        plan = [state["question"]]
    _log(state, f"Planner: {len(plan)} sub-questions.")
    return {"plan": plan}


def search_node(state: AgentState) -> AgentState:
    settings = state["_settings"]
    _log(state, "Search: querying the web for each sub-question...")
    seen: dict[str, Source] = {}
    n = 1
    for sub in state["plan"]:
        for r in web_search(sub, max_results=settings.max_results_per_query):
            url = r["url"]
            if url in seen:
                continue
            seen[url] = Source(n=n, url=url, title=r.get("title", ""), snippet=r.get("snippet", ""))
            n += 1
    sources = [s.model_dump() for s in seen.values()]
    _log(state, f"Search: collected {len(sources)} unique sources.")
    return {"sources": sources}


def reader_node(state: AgentState) -> AgentState:
    index = state["_index"]
    _log(state, "Reader: fetching pages and building the ephemeral RAG index...")
    indexed = 0
    for s in state["sources"]:
        text = fetch_and_extract(s["url"])
        if text:
            chunks = index.add_document(text, source_n=s["n"], url=s["url"])
            indexed += 1 if chunks else 0
    _log(state, f"Reader: indexed {indexed} pages, {index.size} chunks total.")
    return {}


def _context_block(index: EphemeralIndex, queries: list[str], k: int = 4) -> str:
    seen_ids: set[str] = set()
    lines: list[str] = []
    for q in queries:
        for hit in index.retrieve(q, k=k):
            key = f"{hit['source_n']}::{hit['text'][:60]}"
            if key in seen_ids:
                continue
            seen_ids.add(key)
            lines.append(f"[source {hit['source_n']}] ({hit['url']})\n{hit['text']}")
    return "\n\n".join(lines)


def synthesizer_node(state: AgentState) -> AgentState:
    llm = state["_llm"]
    index = state["_index"]
    revision = state.get("revision", 0)
    _log(state, f"Synthesizer: writing the cited report (revision {revision})...")

    context = _context_block(index, [state["question"], *state["plan"]], k=4)
    feedback = state.get("critic_feedback", "")

    system = (
        "You are a meticulous research writer. Write a well-structured Markdown report "
        "that answers the user's question USING ONLY the provided sources. "
        "Every factual sentence MUST end with one or more citations like [1] or [2][3] "
        "referring to the source numbers. Do not invent sources or facts. "
        "After the report, output a JSON block of the atomic factual claims you made.\n\n"
        "Return STRICT JSON: {\"report_md\": string, \"claims\": "
        "[{\"text\": string, \"cited_sources\": [int, ...]}, ...]}."
    )
    user = f"Question: {state['question']}\n\n"
    if feedback:
        user += f"Reviewer feedback to address in this revision:\n{feedback}\n\n"
    user += f"Sources:\n{context if context else '(no source text was retrieved)'}"

    data = llm.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}], node="synthesizer")
    report_md = data.get("report_md", "") if isinstance(data, dict) else ""
    raw_claims = data.get("claims", []) if isinstance(data, dict) else []
    claims = []
    for c in raw_claims:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        cited = [int(x) for x in c.get("cited_sources", []) if str(x).strip().lstrip("-").isdigit()]
        claims.append(Claim(text=c["text"], cited_sources=cited).model_dump())
    _log(state, f"Synthesizer: drafted report with {len(claims)} claims.")
    return {"report_md": report_md, "claims": claims}


def critic_node(state: AgentState) -> AgentState:
    """Fact-check each claim against the sources it cites (and the index)."""
    llm = state["_llm"]
    index = state["_index"]
    _log(state, "Critic/Verifier: fact-checking every claim against its sources...")

    verified: list[dict] = []
    for claim in state["claims"]:
        evidence = index.retrieve(claim["text"], k=4)
        ev_text = "\n\n".join(f"[source {e['source_n']}] {e['text']}" for e in evidence) or "(no evidence found)"
        system = (
            "You are a strict fact-checker. Given a CLAIM and EVIDENCE excerpts from sources, "
            "decide if the evidence supports the claim. "
            "Return STRICT JSON: {\"verdict\": \"supported\"|\"unsupported\"|\"contradicted\", "
            "\"confidence\": number between 0 and 1, \"rationale\": string}. "
            "Use 'unsupported' when the evidence neither confirms nor denies the claim."
        )
        user = f"CLAIM: {claim['text']}\n\nEVIDENCE:\n{ev_text}"
        data = llm.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}], node="critic")
        verdict = str(data.get("verdict", "unsupported")).lower() if isinstance(data, dict) else "unsupported"
        if verdict not in {"supported", "unsupported", "contradicted"}:
            verdict = "unsupported"
        confidence = float(data.get("confidence", 0.0)) if isinstance(data, dict) else 0.0
        rationale = data.get("rationale", "") if isinstance(data, dict) else ""
        c = Claim(
            text=claim["text"],
            cited_sources=claim.get("cited_sources", []),
            verdict=Verdict(verdict),
            confidence=max(0.0, min(1.0, confidence)),
            rationale=rationale,
        )
        verified.append(c.model_dump())

    flagged = [c for c in verified if c["verdict"] != "supported"]
    _log(state, f"Critic/Verifier: {len(flagged)}/{len(verified)} claims need attention.")

    feedback = ""
    if flagged:
        feedback = "These claims are not supported by the sources. Revise or remove them, or add correct citations:\n"
        feedback += "\n".join(f"- {c['text']} (verdict: {c['verdict']})" for c in flagged)
    return {"claims": verified, "critic_feedback": feedback}


def _route_after_critic(state: AgentState) -> str:
    flagged = [c for c in state["claims"] if c["verdict"] != "supported"]
    if flagged and state.get("revision", 0) < MAX_REVISIONS:
        return "revise"
    return "finish"


def _bump_revision(state: AgentState) -> AgentState:
    return {"revision": state.get("revision", 0) + 1}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("reader", reader_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)
    g.add_node("revise", _bump_revision)

    g.set_entry_point("planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "reader")
    g.add_edge("reader", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", _route_after_critic, {"revise": "revise", "finish": END})
    g.add_edge("revise", "synthesizer")
    return g.compile()


def run_research(
    question: str,
    run_id: str,
    settings: Settings | None = None,
    emit: Callable[[str], None] | None = None,
    llm: LLMClient | None = None,
    index: EphemeralIndex | None = None,
) -> RunResult:
    """Execute the full agent loop and return a populated RunResult.

    `llm` and `index` can be injected (e.g. fakes) for offline testing.
    """
    settings = settings or get_settings()
    llm = llm or LLMClient(settings=settings)
    index = index or EphemeralIndex()
    graph = build_graph()

    initial: AgentState = {
        "question": question,
        "_llm": llm,
        "_index": index,
        "_settings": settings,
        "_emit": emit,
        "revision": 0,
        "log": [],
    }
    final: dict[str, Any] = graph.invoke(initial, config={"recursion_limit": 25})

    return RunResult(
        run_id=run_id,
        status="done",
        question=question,
        plan=final.get("plan", []),
        report_md=final.get("report_md", ""),
        sources=[Source(**s) for s in final.get("sources", [])],
        claims=[Claim(**c) for c in final.get("claims", [])],
        usage=llm.tracker.as_dict(),
        log=final.get("log", []),
    )


if __name__ == "__main__":  # quick manual smoke test
    import sys

    q = " ".join(sys.argv[1:]) or "What are the main differences between HTTP/2 and HTTP/3?"
    result = run_research(q, run_id="local-test")
    print(json.dumps(result.model_dump(), indent=2, default=str))
