from pydantic import BaseModel, Field
from typing import Literal, TypedDict


class PipelineState(TypedDict):
    topic: str
    depth: str
    plan: list[str]
    research_notes: list[str]
    report: str


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Topic to research")
    depth: Literal["brief", "detailed"] = "detailed"


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_findings: list[str]
    full_report: str
    sources_consulted: list[str]
    word_count: int
