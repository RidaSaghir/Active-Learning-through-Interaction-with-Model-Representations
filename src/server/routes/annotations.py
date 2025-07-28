from fastapi import APIRouter
from ..schemas import AnnotationRequest
from .. import state

router = APIRouter()

@router.post("/annotate")
def receive_annotation_request(payload: AnnotationRequest):
    state.pending_annotation_filenames = payload.filenames
    print(f"[FastAPI] New samples to annotate: {len(payload.filenames)}")
    return {"status": "ok"}
