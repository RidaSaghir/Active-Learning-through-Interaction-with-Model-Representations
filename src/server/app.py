from fastapi import FastAPI
from .routes import embeddings, metrics, annotations, visualization

app = FastAPI()
app.include_router(embeddings.router)
app.include_router(metrics.router)
app.include_router(annotations.router)
app.include_router(visualization.router)
