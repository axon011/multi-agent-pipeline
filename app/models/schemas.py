from pydantic import BaseModel, Field
from typing import Literal, TypedDict


class PipelineState(TypedDict, total=False):
    topic: str
    depth: str
    plan: list[str]
    research_notes: list[str]
    sources: list[dict]  # list of {"title", "url", "snippet"}
    routing: list[dict]  # per-question routing decisions
    report: str


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Topic to research")
    depth: Literal["brief", "detailed"] = "detailed"


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_findings: list[str]
    full_report: str
    sources: list[Source] = Field(default_factory=list)
    word_count: int
