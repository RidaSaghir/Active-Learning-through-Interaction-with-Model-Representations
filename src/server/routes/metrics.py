from fastapi import APIRouter
from ..schemas import MetricsPayload
from .. import state
from utils.logging_utils import get_logger

router = APIRouter()
log = get_logger("imlvr.api_metrics")

@router.post("/metrics")
def receive_metrics(payload: MetricsPayload):
    state.latest_metrics = payload.dict()
    log.info(f"Metrics | iter={payload.iteration} | acc={payload.accuracy:.4f} | loss={payload.loss:.4f}")
    return {"status": "ok"}

@router.get("/metrics")
def get_latest_metrics():
    if state.latest_metrics:
        return state.latest_metrics
    return {"error": "No metrics received yet"}
