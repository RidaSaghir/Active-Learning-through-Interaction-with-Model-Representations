import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from .. import state
from utils.logging_utils import get_logger

router = APIRouter()
log = get_logger("imlvr.api_vis")



@router.get("/plot")
def plot_embeddings():
    if state.latest_embeddings is None:
        log.warning("Plot requested but no embeddings")
        return {"error": "No embeddings to plot"}

    log.info(f"Plot | iter={state.latest_iteration} | n={state.latest_embeddings.shape[0]}")
    embeddings = state.latest_embeddings.tolist()
    actual_labels = state.latest_actual_labels
    predicted_labels = state.latest_predicted_labels
    filenames = state.latest_filenames

    html = r"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h2>3D Embeddings - Iteration """ + str(state.latest_iteration) + r"""</h2>
                <div id="plot" style="width: 100%; height: 600px;"></div>
                <script>
                    const embeddings = """ + json.dumps(embeddings) + r""";
                    const actualLabels = """ + json.dumps(actual_labels) + r""";
                    const predictedLabels = """ + json.dumps(predicted_labels) + r""";
                    const filenames = """ + json.dumps(filenames) + r""";

                    const x = embeddings.map(e => e[0]);
                    const y = embeddings.map(e => e[1]);
                    const z = embeddings.map(e => e[2]);

                    const hoverTexts = filenames.map((f, i) => 
                        `File: ${f}<br>Actual: ${actualLabels[i]}<br>Predicted: ${predictedLabels[i]}`
                    );
                    
                    const labelToIndex = {};
                    let labelCounter = 0;
                    const colorIndices = actualLabels.map(label => {
                        if (!(label in labelToIndex)) {
                            labelToIndex[label] = labelCounter++;
                        }
                        return labelToIndex[label];
                    });

                    const trace = {
                        x: x,
                        y: y,
                        z: z,
                        mode: 'markers',
                        type: 'scatter3d',
                        text: hoverTexts,
                        hoverinfo: 'text',
                        marker: {
                            size: 4,
                            color: colorIndices,
                            colorscale: 'Viridis',
                            opacity: 0.8
                        }
                    };

                    Plotly.newPlot('plot', [trace]);
                </script>
            </body>
            </html>
            """
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
        "actual_label": state.latest_actual_labels
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

