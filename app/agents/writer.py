from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import PipelineState

WRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional technical writer. Synthesize research notes into a well-structured, readable report with an executive summary, key findings (as bullet points), and a conclusion."),
    ("user", "Topic: {topic}\n\nResearch Notes:\n{notes}\n\nWrite the full structured report."),
])


def run_writer(state: PipelineState) -> PipelineState:
    from app.config import get_llm
    llm = get_llm(temperature=0.5)
    notes_text = "\n".join(state["research_notes"])
    chain = WRITE_PROMPT | llm
    response = chain.invoke({"topic": state["topic"], "notes": notes_text})
    state["report"] = response.content
    return state
