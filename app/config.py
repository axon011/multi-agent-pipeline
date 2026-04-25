import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-5-turbo")

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


def get_claude_llm():
    from langchain_claude_code import ChatClaudeCode
    return ChatClaudeCode()
