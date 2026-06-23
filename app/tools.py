"""External tools: free web search (ddgs) and page reading (httpx + trafilatura).

No API keys required. These are the agent's only window to the outside world.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("deepresearch.tools")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def web_search(query: str, max_results: int = 4) -> list[dict[str, str]]:
    """Return a list of {title, url, snippet} via DuckDuckGo metasearch (no key)."""
    from ddgs import DDGS

    out: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                out.append(
                    {
                        "title": r.get("title", ""),
                        "url": url,
                        "snippet": r.get("body") or r.get("snippet") or "",
                    }
                )
    except Exception as exc:  # network / rate-limit; degrade gracefully
        logger.warning("web_search failed for %r: %s", query, exc)
    return out


def fetch_and_extract(url: str, timeout: float = 15.0, max_chars: int = 12000) -> str:
    """Fetch a URL and extract its main readable text. Empty string on failure."""
    import trafilatura

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": _USER_AGENT}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.warning("fetch failed for %s: %s", url, exc)
        return ""

    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    return text[:max_chars]
