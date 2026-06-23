"""Evaluation harness: scores faithfulness + citation coverage across a question set.

These are RAGAS-style metrics computed directly from the verifier's own per-claim
verdicts (no extra LLM judge needed):
  - faithfulness      = supported claims / total claims
  - citation_coverage = claims with >=1 citation / total claims

Usage:
  python -m eval.run_eval            # uses eval/questions.json
  python -m eval.run_eval "my question"

Requires a live LLM provider (set GROQ_API_KEY or run Ollama).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents import run_research  # noqa: E402
from app.config import get_settings  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "questions.json"
OUT_FILE = ROOT / "eval" / "results.md"


def main() -> int:
    settings = get_settings()
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("No GROQ_API_KEY set. Set one in .env or switch LLM_PROVIDER=ollama.")
        return 1

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    rows, faith_total, cov_total = [], 0.0, 0.0
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        result = run_research(q, run_id=f"eval-{i}")
        faith_total += result.faithfulness
        cov_total += result.citation_coverage
        rows.append(
            {
                "question": q,
                "claims": len(result.claims),
                "faithfulness": result.faithfulness,
                "citation_coverage": result.citation_coverage,
                "cost_usd": result.usage.get("cost_usd", 0),
            }
        )
        print(f"    faithfulness={result.faithfulness:.0%} coverage={result.citation_coverage:.0%}")

    n = max(len(questions), 1)
    lines = [
        "# DeepResearch Eval Results",
        "",
        f"- Model: `{settings.active_model}` ({settings.llm_provider})",
        f"- Mean faithfulness: **{faith_total / n:.0%}**",
        f"- Mean citation coverage: **{cov_total / n:.0%}**",
        "",
        "| # | Question | Claims | Faithfulness | Coverage | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['question'][:60]} | {r['claims']} | "
            f"{r['faithfulness']:.0%} | {r['citation_coverage']:.0%} | ${r['cost_usd']:.4f} |"
        )
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
