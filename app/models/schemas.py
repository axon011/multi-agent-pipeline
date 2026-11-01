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
    use_opus_planner: bool  # if True, planner runs on Claude Opus instead of Sonnet
    # Critic state — None until critic runs. critique_iterations tracks
    # how many writer-repair passes have run so we cap the loop.
    critique: dict
    critique_iterations: int
    enable_critic: bool


# ---------------------------------------------------------------------------
# LLM-facing structured outputs. These are the contracts between agents:
# the planner and critic must return JSON that validates against them
# (see app/structured.py), rather than free text parsed with regexes.
# ---------------------------------------------------------------------------

# Number of planner questions per depth. `detailed` also asks the writer for
# a longer report (see app/agents/writer.py).
QUESTIONS_PER_DEPTH = {"brief": 3, "detailed": 5}


class ResearchPlan(BaseModel):
    """Planner output: the sub-questions the Researcher will answer."""
    questions: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Specific, answerable research questions, one per entry, "
                    "without numbering.",
    )


class ClaimVerdict(BaseModel):
    """Critic output for a single claim."""
    index: int = Field(ge=0, description="The claim index from the input.")
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = Field(default="", description="One short sentence.")


class CritiqueOutput(BaseModel):
    """Critic output: one verdict per audited claim."""
    verdicts: list[ClaimVerdict]


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""


class CritiqueVerdict(BaseModel):
    """Per-claim verdict from the Critic agent."""
    claim: str
    citation_indices: list[int] = Field(default_factory=list)
    verdict: Literal["supported", "partial", "unsupported", "no_citation"]
    reason: str = ""


class CritiqueReport(BaseModel):
    """Aggregate critique of the Writer's report."""
    score: float = Field(ge=0.0, le=1.0)
    verdicts: list[CritiqueVerdict] = Field(default_factory=list)
    repaired: bool = False
    iterations: int = 1

    @property
    def supported_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "supported")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "unsupported")


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Topic to research")
    depth: Literal["brief", "detailed"] = "detailed"
    use_opus_planner: bool = Field(
        default=False,
        description=(
            "Run the planner on Claude Opus instead of Sonnet (Claude CLI "
            "backend only; ignored on OpenAI-compatible backends)."
        ),
    )
    engine: Literal["langgraph", "crew"] = Field(
        default="langgraph",
        description=(
            "Orchestration engine. 'langgraph' (default) uses the typed-state "
            "graph pipeline; 'crew' runs the same 3 roles via CrewAI Agents/Tasks."
        ),
    )
    enable_critic: bool = Field(
        default=True,
        description=(
            "Run the Critic agent after the Writer to verify citations and "
            "trigger one repair pass if grounding score is low."
        ),
    )


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_findings: list[str]
    full_report: str
    sources: list[Source] = Field(default_factory=list)
    word_count: int
    critique: CritiqueReport | None = None
