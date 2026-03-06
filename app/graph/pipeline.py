from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.writer import run_writer
from app.models.schemas import ResearchReport


class PipelineState(TypedDict):
    topic: str
    depth: str
    plan: list[str]
    research_notes: list[str]
    report: str


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("planner", run_planner)
    graph.add_node("researcher", run_researcher)
    graph.add_node("writer", run_writer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


COMPILED_GRAPH = build_graph()


async def run_pipeline(topic: str, depth: str) -> ResearchReport:
    initial_state: PipelineState = {
        "topic": topic,
        "depth": depth,
        "plan": [],
        "research_notes": [],
        "report": "",
    }

    final_state = await COMPILED_GRAPH.ainvoke(initial_state)

    report_text = final_state["report"]
    findings = [line.strip("- ") for line in report_text.split("\n") if line.startswith("-")][:5]

    return ResearchReport(
        topic=topic,
        summary=report_text[:300],
        key_findings=findings or ["See full report"],
        full_report=report_text,
        sources_consulted=final_state.get("research_notes", []),
        word_count=len(report_text.split()),
    )
