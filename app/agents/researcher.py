from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import PipelineState

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a thorough researcher. Answer each research question with factual, detailed information."),
    ("user", "Research Plan:\n{plan}\n\nFor each question, write 2-3 sentences of factual findings."),
])


def run_researcher(state: PipelineState) -> PipelineState:
    from app.config import get_llm
    llm = get_llm(temperature=0.4)
    plan_text = "\n".join(state["plan"])
    chain = RESEARCH_PROMPT | llm
    response = chain.invoke({"plan": plan_text})
    notes = [line.strip() for line in response.content.split("\n") if line.strip()]
    state["research_notes"] = notes
    return state
