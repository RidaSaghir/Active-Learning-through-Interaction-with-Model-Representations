from fastapi import APIRouter
from ..schemas import MetricsPayload
from .. import state
from utils.logging_utils import get_logger

router = APIRouter()
log = get_logger("imlvr.api_metrics")

@router.post("/metrics")
def receive_metrics(payload: MetricsPayload):
    state.latest_metrics = payload.dict()
    state.metrics_history.append(payload.dict())
    log.info(
        f"Metrics | iter={payload.iteration} | phase={payload.phase} | "
        f"acc={payload.accuracy:.4f} | loss={payload.loss:.4f} | "
        f"human_labels={payload.human_labeled}"
    )
    return {"status": "ok"}

@router.get("/metrics")
def get_metrics():
    if getattr(state, "latest_metrics", None) is None:
        return {"error": "No metrics yet"}
    return state.latest_metrics

