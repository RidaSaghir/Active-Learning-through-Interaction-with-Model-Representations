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
def post_human_annotations(payload: HumanAnnotation):
    if os.path.exists(HUMAN_ANNOTATIONS):
        with open(HUMAN_ANNOTATIONS, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    updated = 0
    user = payload.user or "anonymous"

    for filename, idx, label in zip(payload.filenames, payload.indices, payload.labels):
        key = str(int(idx))
        val = int(label)

        prev = annotations.get(key)
        # support old format (int) and new format (dict)
        if isinstance(prev, dict):
            prev_label = prev.get("label")
        else:
            prev_label = prev

        # only update if new or changed
        if prev_label != val:
            annotations[key] = {
                "label": val,
                "user": user,
                "filename": filename,
            }
            updated += 1

    if updated > 0:
        tmp = f"{HUMAN_ANNOTATIONS}.tmp"
        with open(tmp, "w") as f:
            json.dump(annotations, f, indent=2)
        os.replace(tmp, HUMAN_ANNOTATIONS)
        # clear pending suggestions once consumed
        try:
            from .. import state
            state.suggested_annotation_items = []
        except Exception:
            pass
        log.info(f"Stored human annotations | updated={updated} | total={len(annotations)}")
    else:
        log.debug("Human annotations POST contained no new labels.")

    return {"status": "ok", "updated": updated, "total": len(annotations)}


@router.get("/classes")
def get_class_map():
    return getattr(state, "class_code_to_label", {})

