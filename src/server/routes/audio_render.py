from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException
from pathlib import Path
from urllib.parse import unquote
from utils.logging_utils import get_logger
from config import DATA_DIR

router = APIRouter()
log = get_logger("imlvr.api_audio_render")
@router.get("/audio/{filename}")
def serve_audio(filename: str):
    filename = unquote(filename)  # important if any spaces/%xx
    log.info(f"Audio request: filename={filename} DATA_DIR={DATA_DIR}")
    for fold in range(10):
        candidate = Path(DATA_DIR) / f"fold{fold}" / filename
        log.info(f" - check: {candidate}")
        if candidate.exists():
            return FileResponse(
                candidate,
                media_type="audio/wav",
                filename=filename,
                headers={
                    "Content-Disposition": "inline"
                }
            )
    raise HTTPException(status_code=404, detail=f"File {filename} not found in any fold")
