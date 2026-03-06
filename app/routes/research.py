from fastapi import APIRouter, HTTPException
from app.models.schemas import ResearchRequest, ResearchReport
from app.graph.pipeline import run_pipeline

router = APIRouter()


@router.post("/", response_model=ResearchReport)
async def research(request: ResearchRequest):
    try:
        report = await run_pipeline(request.topic, request.depth)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
