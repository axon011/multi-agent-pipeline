"""Citation numbering across research questions.

Each question's LLM synthesis cites its own search results as [1], [2], ...
but the Writer and Critic resolve [N] against the merged state["sources"]
list. These tests pin the renumbering that keeps the two in sync.
"""

import re

import pytest

import app.config
from app.agents import researcher
from app.agents.critic import extract_claims
from app.tools.search import SearchResult


def _result(name: str) -> SearchResult:
    return SearchResult(title=name, url=f"https://example.org/{name}", snippet=f"snippet {name}")


def test_renumber_citations_maps_local_to_global():
    text = "Alpha [1]. Beta [2][1]. Gamma [3]."
    assert researcher._renumber_citations(text, {1: 4, 2: 5, 3: 6}) == (
        "Alpha [4]. Beta [5][4]. Gamma [6]."
    )


def test_renumber_citations_drops_markers_with_no_source():
    # [7] was never shown to the LLM, so there is no source to resolve it to.
    assert researcher._renumber_citations("Claim [1] and [7].", {1: 2}) == "Claim [2] and."


@pytest.mark.asyncio
async def test_citations_resolve_to_the_source_the_llm_actually_saw(monkeypatch):
    q1_results = [_result("q1-a"), _result("q1-b")]
    # q2 shares one URL with q1 (dedup) and has one new result.
    q2_results = [_result("q1-b"), _result("q2-c")]

    async def fake_one_question(question, llm, topic=None):
        if "first" in question:
            return "**first**\nFact from q1-a [1]. Fact from q1-b [2].", q1_results, ["duckduckgo"]
        return "**second**\nFact from q1-b [1]. Fact from q2-c [2].", q2_results, ["duckduckgo"]

    monkeypatch.setattr(researcher, "_research_one_question", fake_one_question)
    monkeypatch.setattr(app.config, "get_llm", lambda temperature=0.3: None)

    state = await researcher._run_researcher_async(
        {"topic": "t", "plan": ["1. first question", "2. second question"]}
    )

    sources = state["sources"]
    assert [s["title"] for s in sources] == ["q1-a", "q1-b", "q2-c"]

    # Every "Fact from X [N]" must point at the source titled X.
    report = "\n".join(state["research_notes"])
    for claim, indices in extract_claims(report):
        expected_title = re.search(r"Fact from (\S+)", claim).group(1)
        assert [sources[i - 1]["title"] for i in indices] == [expected_title], claim
