import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

app = FastAPI()


# ---- Model for incoming POST ----
class EmbeddingPayload(BaseModel):
    iteration: int
    embedding_shape: List[int]
    embeddings: List[List[float]]  # List of vectors
    labels: List[int]
    filenames: List[str]


# ---- Global variable to hold the most recent embedding matrix ----
latest_embeddings = None
latest_iteration = None
latest_labels = None
latest_filenames = None

# ---- POST /embeddings ----
@app.post("/embeddings")
def receive_embeddings(payload: EmbeddingPayload):
    global latest_embeddings, latest_iteration, latest_labels, latest_filenames
    latest_embeddings = np.array(payload.embeddings)
    latest_labels = np.array(payload.labels)
    latest_iteration = payload.iteration
    print(f"[FastAPI] Received embeddings at iteration {latest_iteration}, shape: {latest_embeddings.shape}")
    return {"status": "ok"}


# ---- GET /latest_embeddings ----
@app.get("/latest_embeddings")
def get_latest_embeddings():
    if latest_embeddings is None:
        return {"error": "No embeddings received yet"}

    return {
        "iteration": latest_iteration,
        "shape": list(latest_embeddings.shape),
        "embeddings": latest_embeddings.tolist(),
        "labels": latest_labels.tolist(),
        "filenames": latest_filenames
    }


# ---- GET /plot ----
@app.get("/plot")
def plot_embeddings():
    if latest_embeddings is None:
        return {"error": "No embeddings to plot"}

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latest_embeddings)

    plt.figure(figsize=(6, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(f"PCA of embeddings (Iteration {latest_iteration})")
    plt.savefig("embedding_plot.png")
    plt.close()
    return FileResponse("embedding_plot.png")


# ---- GET /table ----
@app.get("/table", response_class=HTMLResponse)
def get_embedding_table():
    if latest_embeddings is None or latest_labels is None or latest_filenames is None:
        return "<h2>No embeddings, labels, or filenames available yet.</h2>"

    col_names = [f"dim_{i}" for i in range(latest_embeddings.shape[1])]
    df = pd.DataFrame(latest_embeddings, columns=col_names)
    df.insert(0, "filename", latest_filenames)  # 👈 insert as first column
    df["label"] = latest_labels  # 👈 label as last column

    html_table = df.to_html(index=False, classes="table table-striped", border=0)

    html_page = f"""
    <html>
    <head>
        <title>Embedding Table</title>
        <style>
            body {{
                font-family: sans-serif;
                padding: 2rem;
            }}
            .table {{
                border-collapse: collapse;
                width: 100%;
                overflow-x: auto;
            }}
            .table td, .table th {{
                border: 1px solid #ddd;
                padding: 8px;
                font-size: 12px;
            }}
            .table tr:nth-child(even){{background-color: #f2f2f2;}}
            .table th {{
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #4CAF50;
                color: white;
            }}
        </style>
    </head>
    <body>
        <h2>Embeddings Table (Iteration {latest_iteration})</h2>
        <div style="overflow-x:auto;">
            {html_table}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_page)

