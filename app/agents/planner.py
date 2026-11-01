from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import QUESTIONS_PER_DEPTH, PipelineState, ResearchPlan
from app.observability import observe, trace_llm, update_current
from app.structured import invoke_structured

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research planner. Given a topic, produce exactly "
        "{num_questions} specific, answerable research questions that "
        "together cover different facets of the topic. Each question must "
        "stand alone (no 'it' or 'they' referring to other questions) and "
        "must not be numbered.",
    ),
    ("user", "Topic: {topic}"),
])


def plan_size(depth: str) -> int:
    return QUESTIONS_PER_DEPTH.get(depth, QUESTIONS_PER_DEPTH["detailed"])


def clean_plan(plan: ResearchPlan, max_questions: int) -> list[str]:
    """Strip blanks and duplicates, then cap at the requested size."""
    seen: set[str] = set()
    questions: list[str] = []
    for q in plan.questions:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            questions.append(q)
    return questions[:max_questions]


@observe(name="planner", as_type="agent", capture_input=False, capture_output=False)
def run_planner(state: PipelineState) -> PipelineState:
    from app.config import get_planner_llm, planner_model_label

    use_opus = bool(state.get("use_opus_planner"))
    llm = get_planner_llm(use_opus=use_opus)
    num_questions = plan_size(state.get("depth", "detailed"))
    payload = {"topic": state["topic"], "num_questions": num_questions}

    with trace_llm("planner.invoke", model=planner_model_label(use_opus), input=payload):
        plan = invoke_structured(llm, PLAN_PROMPT, payload, ResearchPlan)
        update_current(output=plan.model_dump())

    state["plan"] = clean_plan(plan, num_questions)
    return state
