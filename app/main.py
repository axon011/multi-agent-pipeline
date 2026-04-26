from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.routes import research

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Multi-Agent Research Pipeline",
    description="LangGraph research pipeline with web search and SSE streaming",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router, prefix="/research", tags=["research"])


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
