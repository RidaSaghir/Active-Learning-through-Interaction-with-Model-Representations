import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
from sklearn.decomposition import PCA
from .. import state
from pathlib import Path
from utils.logging_utils import get_logger

router = APIRouter()
log = get_logger("imlvr.api_vis")



@router.get("/plot")
def plot_embeddings():
    if state.latest_embeddings is None:
        log.warning("Plot requested but no embeddings")
        return {"error": "No embeddings to plot"}

    log.info(
        f"Plot | iter={state.latest_iteration} | "
        f"n={state.latest_embeddings.shape[0]}"
    )

    # adjust this path if needed
    template_path = Path(__file__).resolve().parent.parent / "plot.html"
    html_template = template_path.read_text(encoding="utf-8")

    embeddings = state.latest_embeddings.tolist()
    actual_labels = state.latest_actual_labels
    predicted_labels = state.latest_predicted_labels
    filenames = state.latest_filenames
    cues = state.cues or {}
    is_labeled = state.is_labeled or []

    html = (
        html_template
        .replace("{{ITER}}", str(state.latest_iteration))
        .replace("{{EMBEDDINGS}}", json.dumps(embeddings))
        .replace("{{LABELS}}", json.dumps(actual_labels))
        .replace("{{PRED}}", json.dumps(predicted_labels))
        .replace("{{CUES}}", json.dumps(cues))
        .replace("{{FILENAMES}}", json.dumps(filenames))
        .replace("{{ISLABELED}}", json.dumps(is_labeled))
    )

    return HTMLResponse(content=html)


@router.get("/table", response_class=HTMLResponse)
def get_embedding_table():
    if state.latest_embeddings is None:
        return "<h2>No data available.</h2>"

    col_names = [f"dim_{i}" for i in range(state.latest_embeddings.shape[1])]
    df_emb = pd.DataFrame(state.latest_embeddings, columns=col_names)
    df_meta = pd.DataFrame({
        "filename": state.latest_filenames,
        "predicted_label": state.latest_predicted_labels,
        "actual_label": state.latest_actual_labels,
        "is_labeled": state.is_labeled,
    })
    df_cues = pd.DataFrame(state.cues)
    df = pd.concat([df_meta, df_cues, df_emb], axis=1)

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
           <h2>Embeddings Table (Iteration {state.latest_iteration})</h2>
           <div style="overflow-x:auto;">
               {html_table}
           </div>
       </body>
       </html>
       """
    return HTMLResponse(content=html_page)

