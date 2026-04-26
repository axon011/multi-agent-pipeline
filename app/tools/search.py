import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Synchronous DuckDuckGo search. Returns list of SearchResult."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", "") or r.get("url", ""),
                snippet=r.get("body", ""),
            )
            for r in results
        ]
    except Exception as e:
        logger.warning("Web search failed for query %r: %s", query, e)
        return []


async def web_search_async(query: str, max_results: int = 5) -> list[SearchResult]:
    """Async wrapper that runs DuckDuckGo search in a thread."""
    return await asyncio.to_thread(web_search, query, max_results)


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """Format search results as numbered context for an LLM prompt."""
    if not results:
        return "(no search results found)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    {r.snippet}")
    return "\n".join(lines)
