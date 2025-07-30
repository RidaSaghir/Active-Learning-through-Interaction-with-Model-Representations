from fastapi import APIRouter
import numpy as np
from ..schemas import EmbeddingPayload, EmbeddingPayload3D
from .. import state

router = APIRouter()

@router.post("/latest_embeddings")
def receive_latest_embeddings(payload: EmbeddingPayload):
    state.latest_iteration = payload.iteration
    state.latest_embeddings = np.array(payload.embeddings)
    state.latest_actual_labels = payload.actual_labels
    state.latest_predicted_labels = payload.predicted_labels
    state.latest_filenames = payload.filenames

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
        "actual_labels": state.latest_actual_labels,
        "predicted_labels": state.latest_predicted_labels,
        "filenames": state.latest_filenames
    }

@router.post("/3d_embeddings")
def receive_latest_3d_embeddings(payload: EmbeddingPayload3D):
    state.latest_iteration = payload.iteration
    state.latest_3d_embeddings = np.array(payload.embeddings)
    state.latest_actual_labels = payload.actual_labels
    state.latest_predicted_labels = payload.predicted_labels
    state.latest_filenames = payload.filenames

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
        "actual_labels": state.latest_actual_labels,
        "predicted_labels": state.latest_predicted_labels,
        "filenames": state.latest_filenames
    }