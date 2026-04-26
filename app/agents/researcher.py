import asyncio
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import PipelineState
from app.tools.search import (
    SearchResult,
    format_results_for_prompt,
    web_search_async,
)

logger = logging.getLogger(__name__)

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a thorough researcher. Synthesize the provided web search "
        "results into a factual, well-cited answer to the question. "
        "When citing facts, reference sources as [1], [2], etc., matching "
        "the numbered results provided. Stay grounded — do not invent facts "
        "that aren't supported by the sources."
    ),
    (
        "user",
        "Question: {question}\n\n"
        "Web search results:\n{search_context}\n\n"
        "Write a focused 3-5 sentence answer with inline citations like [1], [2]."
    ),
])


def _strip_numbering(line: str) -> str:
    return re.sub(r"^\s*[\d]+[\.\)]\s*", "", line).strip()


async def _research_one_question(question: str, llm) -> tuple[str, list[SearchResult]]:
    """Search the web for a question, then ask the LLM to synthesize an answer."""
    cleaned = _strip_numbering(question)
    results = await web_search_async(cleaned, max_results=4)
    search_context = format_results_for_prompt(results)

    chain = RESEARCH_PROMPT | llm
    response = await chain.ainvoke({
        "question": cleaned,
        "search_context": search_context,
    })

    note = f"**{cleaned}**\n{response.content}"
    return note, results


async def _run_researcher_async(state: PipelineState) -> PipelineState:
    from app.config import get_llm

    llm = get_llm(temperature=0.4)
    questions = [q for q in state.get("plan", []) if q.strip()]

    if not questions:
        state["research_notes"] = []
        state["sources"] = []
        return state

    tasks = [_research_one_question(q, llm) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    notes: list[str] = []
    all_sources: list[dict] = []
    seen_urls: set[str] = set()

    for item in results:
        if isinstance(item, Exception):
            logger.warning("research subtask failed: %s", item)
            continue
        note, sources = item
        notes.append(note)
        for s in sources:
            if s.url and s.url not in seen_urls:
                seen_urls.add(s.url)
                all_sources.append(s.to_dict())

    state["research_notes"] = notes
    state["sources"] = all_sources
    return state


def run_researcher(state: PipelineState) -> PipelineState:
    """LangGraph node — runs async pipeline in a fresh event loop if needed."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (FastAPI / LangGraph async path)
        # Schedule and wait. LangGraph will normally call the async version
        # via `astream`/`ainvoke`, so this branch is a fallback.
        future = asyncio.ensure_future(_run_researcher_async(state))
        return loop.run_until_complete(future)
    return asyncio.run(_run_researcher_async(state))


async def arun_researcher(state: PipelineState) -> PipelineState:
    return await _run_researcher_async(state)
