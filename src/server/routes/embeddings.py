from fastapi import APIRouter
import numpy as np
from ..schemas import EmbeddingPayload
from .. import state

router = APIRouter()

@router.post("/embeddings")
def receive_embeddings(payload: EmbeddingPayload):
    state.latest_embeddings = np.array(payload.embeddings)
    state.latest_labels = np.array(payload.labels)
    state.latest_filenames = list(payload.filenames)
    state.latest_label_types = payload.label_types
    state.latest_iteration = payload.iteration
    print(f"[FastAPI] Received embeddings at iteration {payload.iteration}, shape: {state.latest_embeddings.shape}")
    return {"status": "ok"}

@router.get("/latest_embeddings")
def get_latest_embeddings():
    if state.latest_embeddings is None:
        return {"error": "No embeddings received yet"}

    return {
        "iteration": state.latest_iteration,
        "shape": list(state.latest_embeddings.shape),
        "embeddings": state.latest_embeddings.tolist(),
        "labels": state.latest_labels.tolist(),
        "filenames": state.latest_filenames
    }
