from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import PipelineState

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research planner. Given a topic, produce a numbered list of 3-5 specific research questions to answer. Be concise."),
    ("user", "Topic: {topic}\nDepth: {depth}\n\nProduce the research plan as a numbered list."),
])


def run_planner(state: PipelineState) -> PipelineState:
    from app.config import get_claude_llm
    llm = get_claude_llm()
    chain = PLAN_PROMPT | llm
    response = chain.invoke({"topic": state["topic"], "depth": state["depth"]})
    plan_lines = [line.strip() for line in response.content.split("\n") if line.strip()]
    state["plan"] = plan_lines
    return state
