// ----- Data from backend (injected via window.IMLVR_DATA) -----
const {
    embeddings,
    actualLabels,
    predictedLabels,
    filenames,
    cues,
    isLabeledRaw,
    datasetIndices,
    trueCodes
} = window.IMLVR_DATA;

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

function quantile(normArr, q) {
    const finite = normArr.filter(Number.isFinite);
    if (!finite.length) return NaN;
    const sorted = [...finite].sort((a, b) => a - b);
    const idx = Math.floor(q * (sorted.length - 1));
    return sorted[idx];
}

const UNC_HIGH = quantile(uncertaintyNorm, 0.66);
const DIV_HIGH = quantile(diversityNorm,   0.66);
const NOV_HIGH = quantile(noveltyNorm,     0.66);

function binDiversity(divNorm) {
    const finite = divNorm.filter(Number.isFinite);
    if (!finite.length) {
        return divNorm.map(_ => 1); // everything "medium"
    }
    const sorted = [...finite].sort((a, b) => a - b);
    const q33 = sorted[Math.floor(0.33 * (sorted.length - 1))];
    const q66 = sorted[Math.floor(0.66 * (sorted.length - 1))];

    return divNorm.map(v => {
        if (!Number.isFinite(v)) return 1;
        if (v <= q33) return 0;   // low diversity
        if (v <= q66) return 1;   // medium
        return 2;                 // high
    });
}

const diversityBins = binDiversity(diversityNorm); // 0,1,2
const opacityPerBin = [0.15, 0.45, 0.9];  // low / medium / high

const hoverTextsAll = filenames.map((f, i) =>
    `<b>${f}</b>` +
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

const sizeAll = uncertaintyNorm.map(u => 3 + 17 * (u || 0));  // 3..20

const symbolAll = (function() {
    if (!noveltyNorm.length) return [];
    const finite = noveltyNorm.filter(v => Number.isFinite(v));
    if (!finite.length) return noveltyNorm.map(_ => "circle");
    const sorted = [...finite].sort((a,b) => a - b);
    const q33 = sorted[Math.floor(0.33 * (sorted.length - 1))];
    const q66 = sorted[Math.floor(0.66 * (sorted.length - 1))];
    return noveltyNorm.map(v => {
        if (!Number.isFinite(v)) return "circle";
        if (v <= q33) return "x";      // low novelty
        if (v <= q66) return "square"; // medium
        return "circle";               // high
    });
})();

function pick(arr, idxs) {
    return idxs.map(i => arr[i]);
}

function getBaseVisibleIndices(hideLabeled) {
    const idxs = [];
    for (let i = 0; i < embeddings.length; i++) {
        if (hideLabeled && isLabeled[i]) continue;
        idxs.push(i);
    }
    return idxs;
}

const hideCheckbox    = document.getElementById("hideLabeledCheckbox");
const kInput          = document.getElementById("kInput");
const updateKBtn      = document.getElementById("updateKBtn");
const subsetSelect    = document.getElementById("subsetSelect");
const highUncCheckbox = document.getElementById("highUncCheckbox");
const highDivCheckbox = document.getElementById("highDivCheckbox");
const highNovCheckbox = document.getElementById("highNovCheckbox");
const autoRotateCheckbox = document.getElementById("autoRotateCheckbox");

const preset1Btn = document.getElementById("preset1");
const preset2Btn = document.getElementById("preset2");
const preset3Btn = document.getElementById("preset3");

function computeHotspotIndices(k, values, baseVisibleIdxs) {
    const candidates = [];
    for (let v = 0; v < baseVisibleIdxs.length; v++) {
        const i = baseVisibleIdxs[v];
        const val = values[i];
        if (!Number.isFinite(val)) continue;
        candidates.push({i, val});
    }
    candidates.sort((a, b) => b.val - a.val);  // descending
    const kk = Math.min(k, candidates.length);
    return candidates.slice(0, kk).map(o => o.i);
}

function makeData(hideLabeledFlag, k, subsetMode, filters) {
    const baseVisibleIdxs = getBaseVisibleIndices(hideLabeledFlag);

    const covIdxGlobal = computeHotspotIndices(k, coverageNorm, baseVisibleIdxs);
    const denIdxGlobal = computeHotspotIndices(k, densityNorm, baseVisibleIdxs);
    const covSet = new Set(covIdxGlobal);
    const denSet = new Set(denIdxGlobal);

    let visibleIdxs;
    if (subsetMode === "coverage") {
        visibleIdxs = baseVisibleIdxs.filter(i => covSet.has(i));
    } else if (subsetMode === "density") {
        visibleIdxs = baseVisibleIdxs.filter(i => denSet.has(i));
    } else if (subsetMode === "both") {
        visibleIdxs = baseVisibleIdxs.filter(i => covSet.has(i) && denSet.has(i));
    } else {
        visibleIdxs = baseVisibleIdxs.slice(); // all
    }

    if (filters.onlyHighUnc && Number.isFinite(UNC_HIGH)) {
        visibleIdxs = visibleIdxs.filter(i =>
            Number.isFinite(uncertaintyNorm[i]) && uncertaintyNorm[i] >= UNC_HIGH
        );
    }
    if (filters.onlyHighDiv && Number.isFinite(DIV_HIGH)) {
        visibleIdxs = visibleIdxs.filter(i =>
            Number.isFinite(diversityNorm[i]) && diversityNorm[i] >= DIV_HIGH
        );
    }
    if (filters.onlyHighNov && Number.isFinite(NOV_HIGH)) {
        visibleIdxs = visibleIdxs.filter(i =>
            Number.isFinite(noveltyNorm[i]) && noveltyNorm[i] >= NOV_HIGH
        );
    }

    const traces = [];
    const classLabels = Object.keys(classToIndex);

    for (const cls of classLabels) {
        const clsIdx = classToIndex[cls];

        for (let bin = 0; bin < 3; bin++) {
            const idxsThis = visibleIdxs.filter(i =>
                predictedLabels[i] === cls && diversityBins[i] === bin
            );
            if (!idxsThis.length) continue;

            const lineColors = [];
            const lineWidths = [];

            idxsThis.forEach(i => {
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

            const originalIndices = idxsThis.slice();  // global indices

            traces.push({
                name: `${cls} (div bin ${bin})`,
                x: pick(xAll, idxsThis),
                y: pick(yAll, idxsThis),
                z: pick(zAll, idxsThis),
                mode: 'markers',
                type: 'scatter3d',
                text: pick(hoverTextsAll, idxsThis),
                hoverinfo: 'text',
                customdata: originalIndices,
                marker: {
                    size: pick(sizeAll, idxsThis),
                    color: pick(colorIndexAll, idxsThis),
                    colorscale: colorScale,
                    cmin: 0,
                    cmax: nClasses - 1,
                    opacity: opacityPerBin[bin],
                    symbol: pick(symbolAll, idxsThis),
                    line: {
                        color: lineColors,
                        width: lineWidths
                    },
                    colorbar: {
                        title: 'Predicted class',
                        thickness: 15
                    }
                },
                showscale: (clsIdx === 0 && bin === 0),
                showlegend: false
            });
        }
    }

    return traces;
}

const layout = {
    scene: {
        aspectmode: "cube",
        camera: {
            eye: { x: 1.8, y: 1.8, z: 1.4 }
        }
    },
    legend: {orientation: "h"},
    margin: {l: 0, r: 0, t: 0, b: 0},
    showlegend: false
};

const gd = document.getElementById('plot');
const hoverInfoEl = document.getElementById('hoverInfo');

let angle = 0;
let isSpinning = false;
let spinFrameId = null;
let eventsAttached = false;

function spin() {
    if (!isSpinning) return;
    angle += 0.003;
    const r = 1.8;
    Plotly.relayout(gd, {
        'scene.camera.eye': {
            x: r * Math.cos(angle),
            y: r * Math.sin(angle),
            z: 1.4
        }
    });
    spinFrameId = requestAnimationFrame(spin);
}

function startSpin() {
    if (isSpinning) return;
    isSpinning = true;
    spin();
}

function stopSpin() {
    isSpinning = false;
    if (spinFrameId !== null) {
        cancelAnimationFrame(spinFrameId);
        spinFrameId = null;
    }
}

function attachPlotEvents() {
    if (eventsAttached) return;
    eventsAttached = true;

    // Hover feedback: show currently hovered point
    gd.on('plotly_hover', evt => {
        const pt = evt.points[0];
        const globalIdx = pt.customdata;   // from customdata
        if (globalIdx == null) {
            hoverInfoEl.innerHTML = "";
            return;
        }
        hoverInfoEl.innerHTML =
            `Selected: <strong>${filenames[globalIdx]}</strong>` +
            ` · Pred: ${predictedLabels[globalIdx]}` +
            ` · Unc: ${uncertainty[globalIdx]?.toFixed(2)}` +
            ` · Div: ${diversity[globalIdx]?.toFixed(2)}` +
            ` · Nov: ${novelty[globalIdx]?.toFixed(2)}`;
    });

    gd.on('plotly_unhover', () => {
        hoverInfoEl.innerHTML = "";
    });

    let lastUserName = null;
    gd.on('plotly_click', evt => {
        const pt = evt.points[0];
        const globalIdx = pt.customdata;   // index into embeddings / filenames
        console.log("CLICK", { pt, globalIdx });

        if (globalIdx == null) {
            console.warn("No customdata/globalIdx for clicked point");
            return;
        }

        const dsIdx    = datasetIndices[globalIdx];  // dataset index (combined_df index)
        const fn       = filenames[globalIdx];
        const origStr  = actualLabels[globalIdx];
        const predStr  = predictedLabels[globalIdx];
        const trueCode = trueCodes[globalIdx];

        console.log("ANNOT", { dsIdx, fn, origStr, predStr, trueCode });

        if (!lastUserName) {
            const name = prompt("Annotator name:", "user1");
            if (!name) return;
            lastUserName = name;
        }

        fetch("/human_annotations", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                filenames: [fn],
                indices:   [dsIdx],
                labels:    [trueCode],
                user:      lastUserName
            })
        })
        .then(r => {
            console.log("POST status", r.status);
            return r.json();
        })
        .then(data => {
            console.log("Annotation stored:", data);
            hoverInfoEl.innerHTML =
                `Annotated: <strong>${fn}</strong>` +
                ` · true label=${origStr} (code=${trueCode})` +
                ` · user=${lastUserName}`;
        })
        .catch(err => {
            console.error("Failed to send annotation:", err);
            alert("Failed to send annotation. See console for details.");
        });
    });
}

function updatePlot() {
    const kVal       = parseInt(kInput.value) || 10;
    const hideFlag   = hideCheckbox.checked;
    const subsetMode = subsetSelect.value;
    const filters = {
        onlyHighUnc: highUncCheckbox.checked,
        onlyHighDiv: highDivCheckbox.checked,
        onlyHighNov: highNovCheckbox.checked
    };
    const newData = makeData(hideFlag, kVal, subsetMode, filters);
    Plotly.react(gd, newData, layout, {
        displaylogo: false,
        modeBarButtonsToRemove: [
            'toImage', 'select2d', 'lasso2d',
            'hoverCompareCartesian', 'hoverClosestCartesian',
            'resetCameraLastSave3d'
        ]
    }).then(() => {
        attachPlotEvents();
        if (autoRotateCheckbox.checked) {
            startSpin();
        }
    });
}

// Controls
hideCheckbox.addEventListener('change', updatePlot);
updateKBtn.addEventListener('click', updatePlot);
subsetSelect.addEventListener('change', updatePlot);
highUncCheckbox.addEventListener('change', updatePlot);
highDivCheckbox.addEventListener('change', updatePlot);
highNovCheckbox.addEventListener('change', updatePlot);

autoRotateCheckbox.addEventListener('change', () => {
    if (autoRotateCheckbox.checked) {
        startSpin();
    } else {
        stopSpin();
    }
});

// Presets matching your story
preset1Btn.addEventListener('click', () => {
    kInput.value = 20;
    subsetSelect.value = 'coverage';
    highUncCheckbox.checked = true;
    highDivCheckbox.checked = false;
    highNovCheckbox.checked = false;
    updatePlot();
});

preset2Btn.addEventListener('click', () => {
    kInput.value = 50;
    subsetSelect.value = 'all';
    highUncCheckbox.checked = false;
    highDivCheckbox.checked = true;
    highNovCheckbox.checked = true;
    updatePlot();
});

preset3Btn.addEventListener('click', () => {
    kInput.value = 30;
    subsetSelect.value = 'density';
    highUncCheckbox.checked = false;
    highDivCheckbox.checked = false;
    highNovCheckbox.checked = false;
    updatePlot();
});

// Initial render
updatePlot();
