import pytest
from unittest.mock import AsyncMock, patch
from app.models.schemas import ResearchRequest, ResearchReport
from app.tools.router import route_sources


@pytest.mark.asyncio
async def test_research_request_validation():
    req = ResearchRequest(topic="LangGraph overview", depth="brief")
    assert req.topic == "LangGraph overview"
    assert req.depth == "brief"


@pytest.mark.asyncio
async def test_research_request_empty_topic():
    with pytest.raises(Exception):
        ResearchRequest(topic="", depth="brief")


def test_report_schema():
    report = ResearchReport(
        topic="test",
        summary="short summary",
        key_findings=["finding 1", "finding 2"],
        full_report="full report text here",
        sources_consulted=[],
        word_count=5,
    )
    assert report.word_count == 5
    assert len(report.key_findings) == 2


def test_route_sources_matches_definition_query():
    assert route_sources("What is GraphRAG?") == ["wikipedia", "duckduckgo"]


def test_route_sources_propagates_topic_signals():
    assert route_sources(
        "How do they compare?",
        topic="Best vector database libraries for 2026",
    ) == ["github", "hackernews", "duckduckgo"]


def test_route_sources_uses_redundant_fallbacks():
    assert route_sources("Tell me about penguins") == ["duckduckgo", "wikipedia"]


def test_route_sources_matches_acronym_signals():
    assert route_sources("Summarize SOTA methods") == ["arxiv", "duckduckgo"]
    assert route_sources("Compare SDK options") == ["github", "duckduckgo"]


def test_route_sources_without_forced_web_excludes_duckduckgo():
    assert route_sources("What is GraphRAG?", always_include_web=False) == [
        "wikipedia",
    ]
    assert route_sources("Tell me about penguins", always_include_web=False) == [
        "wikipedia",
    ]
