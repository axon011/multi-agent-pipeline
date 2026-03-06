import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.graph.pipeline import PipelineState

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.3)

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research planner. Given a topic, produce a numbered list of 3-5 specific research questions to answer. Be concise."),
    ("user", "Topic: {topic}\nDepth: {depth}\n\nProduce the research plan as a numbered list."),
])


def run_planner(state: PipelineState) -> PipelineState:
    chain = PLAN_PROMPT | llm
    response = chain.invoke({"topic": state["topic"], "depth": state["depth"]})
    plan_lines = [line.strip() for line in response.content.split("\n") if line.strip()]
    state["plan"] = plan_lines
    return state
