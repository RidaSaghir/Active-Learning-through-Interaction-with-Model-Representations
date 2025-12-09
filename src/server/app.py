from fastapi import FastAPI
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from .routes import embeddings, metrics, annotations, visualization, audio_render


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "routes" / "static"

def create_app() -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=(STATIC_DIR)), name="static")
    app.include_router(embeddings.router)
    app.include_router(metrics.router)
    app.include_router(annotations.router)
    app.include_router(visualization.router)
    app.include_router(audio_render.router)

    # Debug: list all routes at startup
    @app.on_event("startup")
    async def _log_routes():
        print("[FastAPI] Routes:")
        for r in app.routes:
            print(" -", getattr(r, "methods", None), getattr(r, "path", None))

    return app