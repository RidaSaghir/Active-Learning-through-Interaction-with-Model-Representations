from fastapi import APIRouter
import numpy as np
from ..schemas import EmbeddingPayload
from .. import state

router = APIRouter()

@router.post("/latest_embeddings")
def receive_latest_embeddings(payload: EmbeddingPayload):
    state.latest_iteration = payload.iteration
    state.latest_embeddings = np.array(payload.embeddings)
    state.latest_labels = np.array(payload.labels)
    state.latest_filenames = payload.filenames
    state.latest_label_types = payload.label_types

    print(f"[FastAPI] Received embeddings at iteration {payload.iteration}, shape: {state.latest_embeddings.shape}")
    return {"status": "ok"}

@router.get("/latest_embeddings")
def get_latest_embeddings():
    if state.latest_embeddings is None:
        return {"error": "No embeddings received yet"}

    print(f"[FastAPI] Received embeddings at iteration {state.latest_iteration}")
    return {
        "iteration": state.latest_iteration,
        "shape": list(state.latest_embeddings.shape),
        "embeddings": state.latest_embeddings.tolist(),
        "labels": state.latest_labels.tolist(),
        "filenames": state.latest_filenames
    }

@router.post("/3d_embeddings")
def receive_latest_3d_embeddings(payload: EmbeddingPayload):
    state.latest_iteration = payload.iteration
    state.latest_3d_embeddings = np.array(payload.embeddings)
    state.latest_labels = np.array(payload.labels)
    state.latest_filenames = payload.filenames
    state.latest_label_types = payload.label_types

    print(f"[FastAPI] Received 3D embeddings at iteration {payload.iteration}, shape: {state.latest_3d_embeddings.shape}")
    return {"status": "ok"}

@router.get("/3d_embeddings")
def get_latest_3d_embeddings():
    if state.latest_3d_embeddings is None:
        return {"error": "No 3D embeddings received yet"}

    print(f"[FastAPI] Received 3D embeddings at iteration {state.latest_iteration}")
    return {
        "iteration": state.latest_iteration,
        "shape": list(state.latest_3d_embeddings.shape),
        "embeddings": state.latest_3d_embeddings.tolist(),
        "labels": state.latest_labels.tolist(),
        "filenames": state.latest_filenames
    }