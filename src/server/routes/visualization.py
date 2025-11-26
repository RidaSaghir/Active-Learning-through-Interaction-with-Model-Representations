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
    cues = state.cues or {}
    is_labeled = state.is_labeled or []

    html = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h2>3D Embeddings - Iteration """ + str(state.latest_iteration) + r"""</h2>

        <div style="margin-bottom: 0.5rem; font-family: sans-serif; font-size: 14px;">
            <label>
                <input type="checkbox" id="hideLabeledCheckbox" />
                Hide labeled samples
            </label>
            &nbsp;&nbsp;&nbsp;
            <label>
                K (hotspots per cue): 
                <input type="number" id="kInput" value="10" min="1" max="500" style="width: 60px;" />
            </label>
            <button id="updateKBtn">Update hotspots</button>
        </div>

        <div id="plot" style="width: 100%; height: 600px;"></div>

        <div id="classLegend" style="margin-top: 1rem; font-family: sans-serif; font-size: 14px;"></div>

        <div style="margin-top: 1rem; font-family: sans-serif; font-size: 13px;">
            <strong>Visual encodings:</strong><br>
            • <strong>Color</strong>: predicted class (10-class palette, see legend).<br>
            • <strong>Size</strong>: larger markers = higher <em>uncertainty</em>.<br>
            • <strong>Opacity</strong>: opaque = high <em>diversity</em> (far from labeled), faded = redundant.<br>
            • <strong>Symbol (shape)</strong> from <em>novelty</em> (binned):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;◯ circle = high novelty<br>
            &nbsp;&nbsp;&nbsp;&nbsp;□ square = medium novelty<br>
            &nbsp;&nbsp;&nbsp;&nbsp;✕ x = low novelty<br>
            • <strong>Red borders</strong>: top-K <em>coverage</em> points (high class-coverage pressure).<br>
            • <strong>Black borders</strong>: top-K <em>density</em> points (high local density).<br>
            • <strong>Purple borders</strong>: points that are both coverage & density hotspots.
        </div>

        <script>
            // ----- Data from backend -----
            const embeddings = """ + json.dumps(embeddings) + r""";
            const actualLabels = """ + json.dumps(actual_labels) + r""";
            const predictedLabels = """ + json.dumps(predicted_labels) + r""";
            const filenames = """ + json.dumps(filenames) + r""";
            const cues = """ + json.dumps(cues) + r""";
            const isLabeledRaw = """ + json.dumps(is_labeled) + r""";

            // Normalize is_labeled to strict booleans
            const isLabeled = isLabeledRaw.map(v => (v === true || v === "true" || v === 1));

            const xAll = embeddings.map(e => e[0]);
            const yAll = embeddings.map(e => e[1]);
            const zAll = embeddings.map(e => e[2]);

            const uncertainty = cues["uncertainty"] || [];
            const density     = cues["density"]     || [];
            const diversity   = cues["diversity"]   || [];
            const novelty     = cues["novelty"]     || [];
            const coverage    = cues["coverage"]    || [];

            function normalize(arr) {
                if (!arr.length) return arr;
                const finite = arr.filter(v => Number.isFinite(v));
                if (!finite.length) return arr;
                const min = Math.min(...finite);
                const max = Math.max(...finite);
                const range = max - min || 1.0;
                return arr.map(v => (v - min) / range);
            }

            const uncertaintyNorm = normalize(uncertainty);
            const diversityNorm   = normalize(diversity);
            const noveltyNorm     = normalize(novelty);
            const densityNorm     = normalize(density);
            const coverageNorm    = normalize(coverage);

            // Hover text with all cues
            const hoverTextsAll = filenames.map((f, i) =>
                `File: ${f}` +
                `<br>Actual: ${actualLabels[i]}` +
                `<br>Predicted: ${predictedLabels[i]}` +
                `<br>is_labeled: ${isLabeled[i]}` +
                `<br>uncertainty: ${uncertainty[i]?.toFixed(3)}` +
                `<br>density: ${density[i]?.toFixed(3)}` +
                `<br>diversity: ${diversity[i]?.toFixed(3)}` +
                `<br>novelty: ${novelty[i]?.toFixed(3)}` +
                `<br>coverage: ${coverage[i]?.toFixed(3)}`
            );

            // ----- Color by predicted class -----
            const classToIndex = {};
            let classCounter = 0;
            predictedLabels.forEach(lbl => {
                if (!(lbl in classToIndex)) {
                    classToIndex[lbl] = classCounter++;
                }
            });

            const colorIndexAll = predictedLabels.map(lbl => classToIndex[lbl]);

            // 10 distinct colors (extend/repeat if needed)
            const classColors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ];

            const nClasses = Object.keys(classToIndex).length;
            const colorScale = [];
            for (let i = 0; i < nClasses; i++) {
                const t = (nClasses === 1) ? 0.5 : i / (nClasses - 1);
                const color = classColors[i % classColors.length];
                colorScale.push([t, color]);
            }

            // Custom legend for classes (using predicted labels)
            function buildClassLegend() {
                const legendDiv = document.getElementById("classLegend");
                let html = "<strong>Class colors (predicted):</strong><br>";
                const entries = Object.entries(classToIndex).sort((a, b) => a[1] - b[1]);
                entries.forEach(([lbl, idx]) => {
                    const color = classColors[idx % classColors.length];
                    html += `<span style="display:inline-block;width:12px;height:12px;background:${color};border:1px solid #000;margin-right:4px;"></span>`;
                    html += `<span>${lbl}</span><br>`;
                });
                legendDiv.innerHTML = html;
            }
            buildClassLegend();

            // ----- Visual encodings -----

            // Size: from uncertainty
            const sizeAll = uncertaintyNorm.map(u => 3 + 17 * (u || 0));  // 3..20

            // Opacity: from diversity (boost contrast)
            const opacityAll = diversityNorm.map(d => Math.pow(d, 3)); //

            // Symbol: novelty bins (3 bins: circle, square, x)
            const symbolAll = (function() {
                if (!noveltyNorm.length) return [];
                const finite = noveltyNorm.filter(v => Number.isFinite(v));
                if (!finite.length) return noveltyNorm.map(_ => "circle");
                const sorted = [...finite].sort((a,b) => a - b);
                const q33 = sorted[Math.floor(0.33 * (sorted.length - 1))];
                const q66 = sorted[Math.floor(0.66 * (sorted.length - 1))];
                return noveltyNorm.map(v => {
                    if (!Number.isFinite(v)) return "circle";
                    if (v <= q33) return "x";  // low novelty
                    if (v <= q66) return "square";  // medium novelty
                    return "circle";                     // high novelty
                });
            })();

            // ----- Helper to pick by indices -----
            function pick(arr, idxs) {
                return idxs.map(i => arr[i]);
            }

            // Current visibility mask (respecting "hide labeled")
            function getVisibleIndices(hideLabeled) {
                const idxs = [];
                for (let i = 0; i < embeddings.length; i++) {
                    if (hideLabeled && isLabeled[i]) continue;
                    idxs.push(i);
                }
                return idxs;
            }

            const hideCheckbox = document.getElementById("hideLabeledCheckbox");
            const kInput = document.getElementById("kInput");
            const updateKBtn = document.getElementById("updateKBtn");

            function computeHotspotIndices(k, values, visibleIdxs) {
                const candidates = [];
                for (let v = 0; v < visibleIdxs.length; v++) {
                    const i = visibleIdxs[v];
                    const val = values[i];
                    if (!Number.isFinite(val)) continue;
                    candidates.push({i, val});
                }
                candidates.sort((a, b) => b.val - a.val);  // descending
                const kk = Math.min(k, candidates.length);
                return candidates.slice(0, kk).map(o => o.i);
            }

            function makeData(hideLabeledFlag, k) {
                const visibleIdxs = getVisibleIndices(hideLabeledFlag);

                // Hotspots in global index space
                const covIdxGlobal = computeHotspotIndices(k, coverageNorm, visibleIdxs);
                const denIdxGlobal = computeHotspotIndices(k, densityNorm, visibleIdxs);

                const covSet = new Set(covIdxGlobal);
                const denSet = new Set(denIdxGlobal);

                // Build per-visible point border style
                const lineColors = [];
                const lineWidths = [];
                visibleIdxs.forEach(i => {
                    const inCov = covSet.has(i);
                    const inDen = denSet.has(i);
                    if (inCov && inDen) {
                        lineColors.push('purple');
                        lineWidths.push(4);
                    } else if (inCov) {
                        lineColors.push('red');
                        lineWidths.push(3);
                    } else if (inDen) {
                        lineColors.push('black');
                        lineWidths.push(2);
                    } else {
                        lineColors.push('rgba(0,0,0,0)');
                        lineWidths.push(0);
                    }
                });

                const traceBase = {
                    name: 'Samples',
                    x: pick(xAll, visibleIdxs),
                    y: pick(yAll, visibleIdxs),
                    z: pick(zAll, visibleIdxs),
                    mode: 'markers',
                    type: 'scatter3d',
                    text: pick(hoverTextsAll, visibleIdxs),
                    hoverinfo: 'text',
                    marker: {
                        size: pick(sizeAll, visibleIdxs),
                        color: pick(colorIndexAll, visibleIdxs),
                        colorscale: colorScale,
                        cmin: 0,
                        cmax: nClasses - 1,
                        opacity: pick(opacityAll, visibleIdxs),
                        symbol: pick(symbolAll, visibleIdxs),
                        colorbar: {
                            title: 'Predicted class index',
                            thickness: 15
                        },
                        line: {
                            color: lineColors,
                            width: lineWidths
                        }
                    },
                    showscale: true
                };

                return [traceBase];
            }

            // Initial K + hide flag
            const initialK = parseInt(kInput.value) || 10;
            let hideLabeledFlag = hideCheckbox.checked;

            const data = makeData(hideLabeledFlag, initialK);

            const layout = {
                scene: {aspectmode: "cube"},
                legend: {orientation: "h"},
                margin: {l: 0, r: 0, t: 0, b: 0}
            };

            Plotly.newPlot('plot', data, layout).then(gd => {
                function updatePlot() {
                    const kVal = parseInt(kInput.value) || 10;
                    hideLabeledFlag = hideCheckbox.checked;
                    const newData = makeData(hideLabeledFlag, kVal);

                    Plotly.restyle(gd, {
                        x: [newData[0].x],
                        y: [newData[0].y],
                        z: [newData[0].z],
                        'marker.size': [newData[0].marker.size],
                        'marker.color': [newData[0].marker.color],
                        'marker.opacity': [newData[0].marker.opacity],
                        'marker.symbol': [newData[0].marker.symbol],
                        'marker.line.color': [newData[0].marker.line.color],
                        'marker.line.width': [newData[0].marker.line.width]
                    }, [0]);
                }

                hideCheckbox.addEventListener('change', updatePlot);
                updateKBtn.addEventListener('click', updatePlot);
            });
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

