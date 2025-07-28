from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from .. import state

router = APIRouter()

@router.get("/plot")
def plot_embeddings():
    if state.latest_embeddings is None:
        return {"error": "No embeddings to plot"}

    reduced = PCA(n_components=2).fit_transform(state.latest_embeddings)
    plt.figure(figsize=(6, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(f"PCA of embeddings (Iteration {state.latest_iteration})")
    plt.savefig("embedding_plot.png")
    plt.close()
    return FileResponse("embedding_plot.png")

@router.get("/table", response_class=HTMLResponse)
def get_embedding_table():
    if state.latest_embeddings is None:
        return "<h2>No data available.</h2>"

    col_names = [f"dim_{i}" for i in range(state.latest_embeddings.shape[1])]
    df_emb = pd.DataFrame(state.latest_embeddings, columns=col_names)
    df_meta = pd.DataFrame({
        "filename": state.latest_filenames,
        "label": state.latest_labels,
        "label_type": state.latest_label_types
    })
    df = pd.concat([df_meta, df_emb], axis=1)

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

