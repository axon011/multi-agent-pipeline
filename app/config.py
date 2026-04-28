import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-5-turbo")
# "glm" (default) uses Z.ai's GLM-5-turbo for researcher+writer.
# "claude" routes researcher+writer through the Claude CLI subscription
# (free for the user but slower because each call shells out).
LLM_BACKEND = os.getenv("LLM_BACKEND", "glm").lower()


def get_llm(temperature: float = 0.3):
    """Worker LLM (researcher + writer). Backend is env-driven."""
    if LLM_BACKEND == "claude" or not LLM_API_KEY:
        # Fall through to Claude when GLM is unavailable / out of credits.
        return get_claude_llm(model="sonnet")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


def get_claude_llm(model: str = "sonnet"):
    """Claude via CLI/subscription. `model` accepts 'sonnet', 'opus', 'haiku'."""
    from langchain_claude_code import ChatClaudeCode
    return ChatClaudeCode(model=model, permission_mode="default")
