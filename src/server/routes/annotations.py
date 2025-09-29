import os
import json
from fastapi import APIRouter
from ..schemas import AnnotationRequest, HumanAnnotation
from .. import state
from config import HUMAN_ANNOTATIONS
from utils.logging_utils import get_logger

router = APIRouter()
log = get_logger("imlvr.api_annotations")
@router.post("/annotate")
def receive_annotation_request(payload: AnnotationRequest):
    state.suggested_annotation_items = [
        {"index": int(i), "filename": f}
        for f, i in zip(payload.filenames, payload.indices)
    ]
    log.info(f"New annotation request | count={len(state.suggested_annotation_items)}")
    return {"status": "ok"}

@router.get("/annotate")
def get_annotation_requests():
    items = getattr(state, "suggested_annotation_items", None)
    if items:
        return {"items": items}
    return {"error": "No annotation requests yet"}

@router.post("/human_annotations")
def receive_human_annotations(payload: HumanAnnotation):
    # Load existing (flat) map or start empty
    if os.path.exists(HUMAN_ANNOTATIONS):
        with open(HUMAN_ANNOTATIONS, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    # Update flat mapping: idx -> label (ints as strings in JSON)
    for filename, idx, label in zip(payload.filenames, payload.indices, payload.labels):
        annotations[str(int(idx))] = int(label)

    # Atomic write to avoid partial files
    tmp = f"{HUMAN_ANNOTATIONS}.tmp"
    with open(tmp, "w") as f:
        json.dump(annotations, f, indent=2)
    os.replace(tmp, HUMAN_ANNOTATIONS)

    log.info(f"Stored human annotations | updated={len(payload.indices)} | total={len(annotations)}")
    # clear suggestions now that labels arrived
    #state.suggested_annotation_items = []
    return {"status": "ok", "updated": len(payload.indices)}

@router.get("/classes")
def get_class_map():
    return getattr(state, "class_code_to_label", {})

