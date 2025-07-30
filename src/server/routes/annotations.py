import os
import json
from fastapi import APIRouter
from ..schemas import AnnotationRequest, HumanAnnotation
from .. import state
from config import HUMAN_ANNOTATIONS

router = APIRouter()

@router.post("/annotate")
def receive_annotation_request(payload: AnnotationRequest):
    state.suggested_annotation_filenames = payload.filenames
    print(f"[FastAPI] New samples to annotate: {len(payload.filenames)}")
    return {"status": "ok"}

@router.get("/annotate")
def get_annotation_requests():
    if state.suggested_annotation_filenames:
        return state.suggested_annotation_filenames
    return {"error": "No metrics received yet"}

@router.post("/human_annotations")
def receive_human_annotations(payload: HumanAnnotation):
    if os.path.exists(HUMAN_ANNOTATIONS):
        with open(HUMAN_ANNOTATIONS, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    for filename, idx, label in zip(payload.filenames, payload.indices, payload.labels):
        annotations[str(idx)] = {
            "label": label,
            "filename": filename
        }

    with open(HUMAN_ANNOTATIONS, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"[FastAPI] Stored {len(payload.indices)} human annotations.")
    return {"status": "ok", "updated": len(payload.indices)}