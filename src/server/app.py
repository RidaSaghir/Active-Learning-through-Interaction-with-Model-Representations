from fastapi import FastAPI
from .routes import embeddings, metrics, annotations, visualization

def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(embeddings.router)
    app.include_router(metrics.router)
    app.include_router(annotations.router)
    app.include_router(visualization.router)

    # Debug: list all routes at startup
    @app.on_event("startup")
    async def _log_routes():
        print("[FastAPI] Routes:")
        for r in app.routes:
            print(" -", getattr(r, "methods", None), getattr(r, "path", None))

    return app