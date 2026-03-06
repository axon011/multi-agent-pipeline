from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import research

app = FastAPI(
    title="Multi-Agent Research Pipeline",
    description="LangGraph + CrewAI research and report generation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router, prefix="/research", tags=["research"])


@app.get("/health")
def health():
    return {"status": "ok"}
