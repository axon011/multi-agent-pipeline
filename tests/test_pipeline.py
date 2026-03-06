import pytest
from unittest.mock import AsyncMock, patch
from app.models.schemas import ResearchRequest, ResearchReport


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
