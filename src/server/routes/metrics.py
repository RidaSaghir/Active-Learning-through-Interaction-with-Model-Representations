from fastapi import APIRouter
from ..schemas import MetricsPayload
from .. import state

router = APIRouter()

@router.post("/metrics")
def receive_metrics(payload: MetricsPayload):
    state.latest_metrics = payload.dict()
    print(f"[FastAPI] Received metrics at iteration {payload.iteration}")
    return {"status": "ok"}
