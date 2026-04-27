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
        report = await run_pipeline(request.topic, request.depth)
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
    }

    # Map each event type to the node that owns it
    OWNER = {
        "plan": "planner",
        "sources": "researcher",
        "research": "researcher",
        "routing": "researcher",
    }

    async def event_generator():
        yield _sse("start", {"topic": request.topic, "depth": request.depth})
        merged_state: dict = dict(initial_state)

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
