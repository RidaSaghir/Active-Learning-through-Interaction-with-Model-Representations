from fastapi import APIRouter
import numpy as np
from ..schemas import EmbeddingPayload
from .. import state
from utils.logging_utils import get_logger

router = APIRouter()
log = get_logger("imlvr.api_embeddings")
@router.post("/latest_embeddings")
def post_latest_embeddings(payload: EmbeddingPayload):
    arr = np.array(payload.embeddings)
    if arr.ndim != 2 or arr.shape[1] != 3:   # expect 3D now
        return {"error": f"Expected (N,3) embeddings, got {arr.shape}"}
    state.latest_iteration = payload.iteration
    state.latest_embeddings = arr
    state.latest_actual_labels = payload.actual_labels
    state.latest_predicted_labels = payload.predicted_labels
    state.latest_filenames = payload.filenames
    state.indices = payload.indices
    state.is_labeled = payload.is_labeled
    state.cues = payload.cues
    log.info(
        f"Received embeddings | iter={payload.iteration} | "
        f"shape={tuple(payload.embedding_shape)} | "
        f"cues={list(payload.cues.keys())}"
    )
    return {"status": "ok"}

@router.get("/latest_embeddings")
def get_latest_embeddings():
    if state.latest_embeddings is None:
        return {"error": "No embeddings received yet"}

    log.info(f"Served embeddings | iter={state.latest_iteration} | n={state.latest_embeddings.shape[0]}")
    return {
        "iteration": state.latest_iteration,
        "shape": list(state.latest_embeddings.shape),
        "embeddings": state.latest_embeddings.tolist(),
        "actual_labels": state.latest_actual_labels,
        "predicted_labels": state.latest_predicted_labels,
        "filenames": state.latest_filenames,
        "indices": state.indices,
        "is_labeled": state.is_labeled,
        "cues": state.cues,
    }
