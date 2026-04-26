from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import PipelineState

WRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional technical writer. Synthesize the provided "
        "research notes into a well-structured markdown report.\n\n"
        "Use this exact structure:\n"
        "## Executive Summary\n"
        "(2-3 sentence overview)\n\n"
        "## Key Findings\n"
        "- finding 1 (with [1] style citations from the research notes)\n"
        "- finding 2\n"
        "- finding 3\n"
        "- finding 4\n"
        "- finding 5\n\n"
        "## Detailed Analysis\n"
        "(2-3 paragraphs synthesizing the research)\n\n"
        "## Conclusion\n"
        "(1 paragraph)\n\n"
        "Preserve any inline citations like [1], [2] from the research notes — "
        "do not invent new ones. Stay grounded in the provided notes; do not "
        "add facts that aren't supported by them."
    ),
    (
        "user",
        "Topic: {topic}\n\nResearch Notes:\n{notes}\n\n"
        "Write the full structured report following the exact format above."
    ),
])


def run_writer(state: PipelineState) -> PipelineState:
    from app.config import get_llm

    llm = get_llm(temperature=0.5)
    notes_text = "\n\n".join(state.get("research_notes", []))
    chain = WRITE_PROMPT | llm
    response = chain.invoke({"topic": state["topic"], "notes": notes_text})
    state["report"] = response.content
    return state
