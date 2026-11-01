"""Tests for schema-validated LLM output (app/structured.py) and the
planner/critic agents that depend on it. No network: LLMs are fakes."""
import json
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.agents.critic import CRITIC_PROMPT, _critique_async
from app.agents.planner import PLAN_PROMPT, clean_plan, run_planner
from app.crew.pipeline import _parse_plan_output
from app.models.schemas import CritiqueOutput, ResearchPlan
from app.structured import (
    StructuredOutputError,
    extract_json,
    invoke_structured,
    parse_model,
)

PLAN_PAYLOAD = {"topic": "GraphRAG", "num_questions": 3}


# --- extract_json / parse_model -------------------------------------------

def test_extract_json_strips_fences_and_prose():
    raw = 'Sure! Here is the plan:\n```json\n{"questions": ["a?"]}\n```\nHope that helps.'
    assert json.loads(extract_json(raw)) == {"questions": ["a?"]}


def test_parse_model_validates_schema():
    plan = parse_model('{"questions": ["What is GraphRAG?"]}', ResearchPlan)
    assert plan.questions == ["What is GraphRAG?"]


@pytest.mark.parametrize("raw", ["not json", '{"questions": []}', '{"qs": ["a"]}'])
def test_parse_model_raises_on_invalid(raw):
    with pytest.raises(StructuredOutputError):
        parse_model(raw, ResearchPlan)


# --- invoke_structured: JSON path (Claude CLI backend) --------------------

def test_invoke_structured_json_path():
    llm = FakeListChatModel(responses=['{"questions": ["q1?", "q2?"]}'])
    plan = invoke_structured(llm, PLAN_PROMPT, PLAN_PAYLOAD, ResearchPlan)
    assert plan.questions == ["q1?", "q2?"]


def test_invoke_structured_retries_once_with_validation_error():
    llm = FakeListChatModel(
        responses=["1. q1?\n2. q2?", '{"questions": ["q1?", "q2?"]}']
    )
    plan = invoke_structured(llm, PLAN_PROMPT, PLAN_PAYLOAD, ResearchPlan)
    assert plan.questions == ["q1?", "q2?"]


def test_invoke_structured_raises_after_retries_exhausted():
    llm = FakeListChatModel(responses=["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, PLAN_PROMPT, PLAN_PAYLOAD, ResearchPlan)


# --- invoke_structured: native path (OpenAI-compatible backend) -----------

def test_invoke_structured_uses_function_calling_for_openai_models():
    llm = ChatOpenAI(model="glm-5-turbo", api_key="test", base_url="http://localhost")
    calls = {}

    def fake_with_structured_output(schema, method):
        calls["schema"], calls["method"] = schema, method
        return RunnableLambda(lambda _: ResearchPlan(questions=["native?"]))

    with patch.object(ChatOpenAI, "with_structured_output", side_effect=fake_with_structured_output):
        plan = invoke_structured(llm, PLAN_PROMPT, PLAN_PAYLOAD, ResearchPlan)

    assert plan.questions == ["native?"]
    assert calls == {"schema": ResearchPlan, "method": "function_calling"}


# --- planner ---------------------------------------------------------------

def test_clean_plan_dedupes_and_caps():
    plan = ResearchPlan(questions=["A?", " a? ", "B?", "", "C?", "D?"])
    assert clean_plan(plan, 3) == ["A?", "B?", "C?"]


@pytest.mark.parametrize("depth,expected", [("brief", 3), ("detailed", 5)])
def test_run_planner_question_count_follows_depth(depth, expected):
    questions = [f"Question {i}?" for i in range(8)]
    llm = FakeListChatModel(responses=[json.dumps({"questions": questions})])
    with patch("app.config.get_planner_llm", return_value=llm):
        state = run_planner({"topic": "GraphRAG", "depth": depth})
    assert state["plan"] == questions[:expected]


def test_crew_plan_parser_validates_json_and_falls_back_to_lines():
    assert _parse_plan_output('{"questions": ["a?", "b?"]}', "brief") == ["a?", "b?"]
    assert _parse_plan_output("# Plan\nfirst?\nsecond?", "brief") == ["first?", "second?"]


# --- critic ----------------------------------------------------------------

SOURCES = [
    {"title": "S1", "url": "https://a", "snippet": "LangGraph is a graph library."},
    {"title": "S2", "url": "https://b", "snippet": "Unrelated text."},
]
REPORT = "LangGraph builds graphs [1]. It was released in 1990 [2]."


@pytest.mark.asyncio
async def test_critique_scores_structured_verdicts():
    reply = CritiqueOutput.model_validate({
        "verdicts": [
            {"index": 0, "verdict": "supported", "reason": "snippet matches"},
            {"index": 1, "verdict": "unsupported", "reason": "no date in snippet"},
        ]
    }).model_dump_json()
    critique = await _critique_async(REPORT, SOURCES, FakeListChatModel(responses=[reply]))
    assert [v.verdict for v in critique.verdicts] == ["supported", "unsupported"]
    assert critique.score == 0.5


@pytest.mark.asyncio
async def test_critique_degrades_to_partial_when_output_never_validates():
    llm = FakeListChatModel(responses=["garbage", "more garbage"])
    critique = await _critique_async(REPORT, SOURCES, llm)
    assert [v.verdict for v in critique.verdicts] == ["partial", "partial"]
    assert critique.score == 0.5


def test_critic_prompt_has_no_freeform_json_instructions():
    # Output format now comes from the schema, not prose in the prompt.
    system = CRITIC_PROMPT.messages[0].prompt.template
    assert "JSON array" not in system
