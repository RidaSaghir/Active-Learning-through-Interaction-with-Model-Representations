from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException
from pathlib import Path
from config import DATA_DIR

router = APIRouter()

@router.get("/audio/{filename}")
def serve_audio(filename: str):
    # Search all folds for the file
    for fold in range(10):
        candidate = Path(DATA_DIR) / f"fold{fold}" / filename
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
