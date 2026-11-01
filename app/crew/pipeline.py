"""CrewAI engine — alternate orchestration of the same 3 roles.

Mirrors the LangGraph engine's behaviour:
    Planner → (multi-source per-question router + fan-out) → Writer
returning the same `ResearchReport` schema so the HTTP route doesn't care
which engine produced the result.

Why not put the multi-source search inside a CrewAI Tool?
    Because the routing/fan-out lives in `app/agents/researcher.py` and is
    shared with the LangGraph engine. Pulling it through CrewAI Tools would
    duplicate logic and we'd lose the topic-aware routing tests we already
    have. The pragmatic shape: CrewAI orchestrates Planner and Writer as
    proper Agents/Tasks; the research fan-out reuses the same async code
    path the LangGraph engine uses.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

# Disable CrewAI's outbound telemetry before import — noisy on networks
# that block telemetry.crewai.com (and irrelevant to us).
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, Process, Task

from app.agents.planner import clean_plan, plan_size
from app.agents.researcher import _run_researcher_async
from app.crew.llm import ClaudeCodeCrewLLM
from app.graph.pipeline import state_to_report
from app.models.schemas import PipelineState, ResearchPlan, ResearchReport
from app.observability import observe, trace_llm, update_current, update_current_trace
from app.structured import StructuredOutputError, parse_model

logger = logging.getLogger(__name__)


def _build_planner_crew(use_opus: bool) -> Crew:
    llm = ClaudeCodeCrewLLM(model="opus" if use_opus else "sonnet")
    planner = Agent(
        role="Research Planner",
        goal="Break a topic into specific, answerable research questions",
        backstory=(
            "You are a thorough research planner. You produce tight, "
            "answerable questions that cover different facets of the topic, "
            "not broad themes."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    plan_task = Task(
        description=(
            "Topic: {topic}\n\n"
            "Produce exactly {num_questions} specific research questions "
            "that, if answered, would let someone write a focused report "
            "on the topic. Each question must stand alone and must not be "
            "numbered.\n\n"
            "Respond with ONLY a JSON object that validates against this "
            "JSON Schema, no prose and no markdown fences:\n"
            # CrewAI only interpolates {identifier} placeholders, so the
            # schema's JSON braces pass through untouched.
            + json.dumps(ResearchPlan.model_json_schema())
        ),
        expected_output='A JSON object of the form {"questions": ["...", "..."]}.',
        agent=planner,
    )
    return Crew(
        agents=[planner],
        tasks=[plan_task],
        process=Process.sequential,
        verbose=False,
    )


def _build_writer_crew(use_opus: bool) -> Crew:
    # Writer always uses Sonnet — Opus toggle is planner-only by design.
    llm = ClaudeCodeCrewLLM(model="sonnet")
    writer = Agent(
        role="Technical Writer",
        goal="Synthesize cited research notes into a structured markdown report",
        backstory=(
            "You are a senior technical writer. You preserve inline citations "
            "like [1], [2] from the research notes and never invent new "
            "facts beyond what the notes support."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    write_task = Task(
        description=(
            "Topic: {topic}\n\n"
            "Research Notes:\n{notes}\n\n"
            "Write a full structured markdown report using EXACTLY this layout:\n"
            "## Executive Summary\n(2-3 sentence overview)\n\n"
            "## Key Findings\n- finding 1 (preserve [N] citations from notes)\n"
            "- finding 2\n- finding 3\n- finding 4\n- finding 5\n\n"
            "## Detailed Analysis\n(2-3 synthesised paragraphs)\n\n"
            "## Conclusion\n(1 paragraph)\n\n"
            "Stay grounded in the provided notes. Do not add facts the notes "
            "do not support."
        ),
        expected_output="A complete structured markdown report with citations.",
        agent=writer,
    )
    return Crew(
        agents=[writer],
        tasks=[write_task],
        process=Process.sequential,
        verbose=False,
    )


def _parse_plan_output(raw: str, depth: str = "detailed") -> list[str]:
    """Validate the planner crew's reply against ``ResearchPlan``.

    CrewAI's custom-LLM path has no native structured output, so we ask for
    JSON in the task and validate it here. If the model ignores the format
    we fall back to treating each non-empty line as a question, so a
    formatting slip degrades the plan rather than failing the run.
    """
    max_questions = plan_size(depth)
    try:
        return clean_plan(parse_model(raw or "", ResearchPlan), max_questions)
    except StructuredOutputError:
        logger.warning("crew planner output failed ResearchPlan validation; line-splitting")
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith(("#", "**Research"))]
    return lines[:max_questions]


@observe(name="run_pipeline.crew", as_type="chain", capture_input=False, capture_output=False)
async def run_crew_pipeline(
    topic: str,
    depth: str,
    use_opus_planner: bool = False,
    enable_critic: bool = True,
) -> ResearchReport:
    """Same contract as `run_pipeline` — engine-swap should be invisible."""
    update_current_trace(
        name="research-pipeline",
        input={
            "topic": topic,
            "depth": depth,
            "use_opus_planner": use_opus_planner,
            "enable_critic": enable_critic,
        },
        metadata={"engine": "crew"},
        tags=["crewai", "research"],
    )

    # Phase 1 — Planner crew (CrewAI). Run sync invoke in a thread because
    # ChatClaudeCode shells out a subprocess and uvicorn's running loop
    # can't host another loop.
    planner_crew = _build_planner_crew(use_opus_planner)
    # The crew engine always runs on the Claude CLI (see app/crew/llm.py).
    model_label = f"claude-cli:{'opus' if use_opus_planner else 'sonnet'}"
    planner_inputs = {"topic": topic, "num_questions": plan_size(depth)}
    with trace_llm(
        "crew.planner.kickoff",
        model=model_label,
        input=planner_inputs,
    ):
        plan_result = await asyncio.to_thread(planner_crew.kickoff, planner_inputs)
        update_current(output=getattr(plan_result, "raw", str(plan_result)))
    plan = _parse_plan_output(getattr(plan_result, "raw", str(plan_result)), depth)
    logger.info("crew planner produced %d questions", len(plan))

    # Phase 2 — Multi-source research (shared with LangGraph engine).
    state: PipelineState = {
        "topic": topic,
        "depth": depth,
        "plan": plan,
        "research_notes": [],
        "sources": [],
        "routing": [],
        "report": "",
        "use_opus_planner": use_opus_planner,
        "enable_critic": enable_critic,
        "critique_iterations": 0,
    }
    state = await _run_researcher_async(state)

    # Phase 3 — Writer crew (CrewAI).
    notes_text = "\n\n".join(state.get("research_notes", []) or [])
    if not notes_text.strip():
        # Edge case: research returned nothing (rate limit, network, etc.).
        # Skip the writer crew so we don't waste a Claude call writing
        # "no notes were provided" boilerplate.
        state["report"] = "_(research phase returned no notes)_"
    else:
        writer_crew = _build_writer_crew(use_opus_planner)
        with trace_llm(
            "crew.writer.kickoff",
            model="claude-cli:sonnet",
            input={"topic": topic, "note_count": len(state.get("research_notes", []))},
        ):
            write_result = await asyncio.to_thread(
                writer_crew.kickoff, {"topic": topic, "notes": notes_text}
            )
            update_current(output=getattr(write_result, "raw", str(write_result)))
        state["report"] = getattr(write_result, "raw", str(write_result))

    # Phase 4 — Critic + optional one-shot repair (shared with LangGraph engine).
    if enable_critic and state.get("report", "").strip():
        from app.agents.critic import (
            REPAIR_THRESHOLD,
            _critique_async,
            repair_report,
        )
        from app.config import get_llm

        critic_llm = get_llm(temperature=0.1)
        critique = await _critique_async(
            state["report"], state.get("sources", []) or [], critic_llm
        )
        state["critique"] = critique.model_dump()
        has_unsupported = any(
            v.verdict == "unsupported" for v in critique.verdicts
        )
        if critique.score < REPAIR_THRESHOLD and has_unsupported:
            repair_llm = get_llm(temperature=0.4)
            revised = await repair_report(
                state["report"],
                critique,
                state.get("sources", []) or [],
                repair_llm,
            )
            state["report"] = revised
            rechecked = await _critique_async(
                revised, state.get("sources", []) or [], critic_llm
            )
            rechecked.repaired = True
            rechecked.iterations = 2
            state["critique"] = rechecked.model_dump()
            state["critique_iterations"] = 1

    report = state_to_report(topic, state)
    update_current_trace(
        output={
            "word_count": report.word_count,
            "source_count": len(report.sources),
            "critique_score": report.critique.score if report.critique else None,
        },
    )
    return report
