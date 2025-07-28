from fastapi import APIRouter
from ..schemas import MetricsPayload
from .. import state

router = APIRouter()

@router.post("/metrics")
def receive_metrics(payload: MetricsPayload):
    state.latest_metrics = payload.dict()
    print(f"[FastAPI] Received metrics at iteration {payload.iteration}")
    return {"status": "ok"}

@router.get("/metrics")
def get_latest_metrics():
    if state.latest_metrics:
        return state.latest_metrics
    return {"error": "No metrics received yet"}
