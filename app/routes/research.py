import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.graph.pipeline import COMPILED_GRAPH, run_pipeline, state_to_report
from app.models.schemas import PipelineState, ResearchRequest, ResearchReport

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ResearchReport)
async def research(request: ResearchRequest):
    """Synchronous endpoint — waits for the full pipeline, returns final report."""
    try:
        if request.engine == "crew":
            from app.crew.pipeline import run_crew_pipeline

            report = await run_crew_pipeline(
                request.topic,
                request.depth,
                use_opus_planner=request.use_opus_planner,
            )
        else:
            report = await run_pipeline(
                request.topic,
                request.depth,
                use_opus_planner=request.use_opus_planner,
            )
        return report
    except Exception as e:
        logger.exception("research pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


STAGE_LABELS = {
    "planner": "Planning research questions",
    "researcher": "Searching the web and synthesizing findings",
    "writer": "Drafting the final report",
}


@router.post("/stream")
async def research_stream(request: ResearchRequest):
    """SSE endpoint — streams stage transitions and the final report."""
    initial_state: PipelineState = {
        "topic": request.topic,
        "depth": request.depth,
        "plan": [],
        "research_notes": [],
        "sources": [],
        "routing": [],
        "report": "",
        "use_opus_planner": request.use_opus_planner,
    }

    # Map each event type to the node that owns it
    OWNER = {
        "plan": "planner",
        "sources": "researcher",
        "research": "researcher",
        "routing": "researcher",
    }

    async def event_generator():
        yield _sse(
            "start",
            {
                "topic": request.topic,
                "depth": request.depth,
                "planner_model": "opus" if request.use_opus_planner else "sonnet",
                "engine": request.engine,
            },
        )
        merged_state: dict = dict(initial_state)

        # CrewAI engine — emit the same event shapes by orchestrating the
        # phases manually. CrewAI's Crew.kickoff doesn't stream per-node
        # deltas the way LangGraph's astream does, so we run each phase
        # ourselves and yield equivalent stage/plan/sources/routing/research
        # events around it. Keeps the SSE contract identical.
        if request.engine == "crew":
            try:
                from app.crew.pipeline import (
                    _build_planner_crew,
                    _build_writer_crew,
                    _parse_plan_output,
                )
                from app.agents.researcher import _run_researcher_async
                import asyncio as _aio

                # Phase 1 — Planner
                yield _sse(
                    "stage",
                    {"node": "planner", "label": STAGE_LABELS["planner"]},
                )
                planner_crew = _build_planner_crew(request.use_opus_planner)
                plan_result = await _aio.to_thread(
                    planner_crew.kickoff,
                    {"topic": request.topic, "depth": request.depth},
                )
                plan = _parse_plan_output(
                    getattr(plan_result, "raw", str(plan_result))
                )
                merged_state["plan"] = plan
                yield _sse("plan", {"questions": plan, "count": len(plan)})

                # Phase 2 — shared researcher
                yield _sse(
                    "stage",
                    {"node": "researcher", "label": STAGE_LABELS["researcher"]},
                )
                researched = await _run_researcher_async(dict(merged_state))
                merged_state.update(researched)
                if merged_state.get("sources"):
                    yield _sse(
                        "sources",
                        {
                            "found": len(merged_state["sources"]),
                            "preview": [
                                {"title": s.get("title", ""), "url": s.get("url", "")}
                                for s in merged_state["sources"][:5]
                            ],
                        },
                    )
                if merged_state.get("routing"):
                    yield _sse("routing", {"per_question": merged_state["routing"]})
                if merged_state.get("research_notes"):
                    yield _sse(
                        "research",
                        {"notes_count": len(merged_state["research_notes"])},
                    )

                # Phase 3 — Writer
                yield _sse(
                    "stage",
                    {"node": "writer", "label": STAGE_LABELS["writer"]},
                )
                notes_text = "\n\n".join(merged_state.get("research_notes", []) or [])
                if not notes_text.strip():
                    merged_state["report"] = "_(research phase returned no notes)_"
                else:
                    writer_crew = _build_writer_crew(request.use_opus_planner)
                    write_result = await _aio.to_thread(
                        writer_crew.kickoff,
                        {"topic": request.topic, "notes": notes_text},
                    )
                    merged_state["report"] = getattr(
                        write_result, "raw", str(write_result)
                    )

                report = state_to_report(request.topic, merged_state)
                yield _sse("complete", report.model_dump())
            except Exception as e:
                logger.exception("crew streaming pipeline failed")
                yield _sse("error", {"message": str(e)})
            return

        try:
            async for chunk in COMPILED_GRAPH.astream(
                initial_state, stream_mode="updates"
            ):
                for node_name, delta in chunk.items():
                    logger.info("LangGraph chunk: node=%s keys=%s", node_name, list((delta or {}).keys()))
                    yield _sse(
                        "stage",
                        {
                            "node": node_name,
                            "label": STAGE_LABELS.get(
                                node_name, node_name.replace("_", " ").title()
                            ),
                        },
                    )

                    if not delta:
                        continue

                    merged_state.update(delta)

                    # Use merged_state since LangGraph's update delta sometimes
                    # omits keys even when the node wrote them.
                    src = merged_state

                    if node_name == OWNER["plan"] and src.get("plan"):
                        yield _sse(
                            "plan",
                            {
                                "questions": src["plan"],
                                "count": len(src["plan"]),
                            },
                        )

                    if node_name == OWNER["sources"] and src.get("sources"):
                        yield _sse(
                            "sources",
                            {
                                "found": len(src["sources"]),
                                "preview": [
                                    {
                                        "title": s.get("title", ""),
                                        "url": s.get("url", ""),
                                    }
                                    for s in src["sources"][:5]
                                ],
                            },
                        )

                    if node_name == OWNER["routing"] and src.get("routing"):
                        yield _sse(
                            "routing",
                            {"per_question": src["routing"]},
                        )

                    if node_name == OWNER["research"] and src.get("research_notes"):
                        yield _sse(
                            "research",
                            {"notes_count": len(src["research_notes"])},
                        )

            report = state_to_report(request.topic, merged_state)
            yield _sse("complete", report.model_dump())

        except Exception as e:
            logger.exception("streaming pipeline failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
