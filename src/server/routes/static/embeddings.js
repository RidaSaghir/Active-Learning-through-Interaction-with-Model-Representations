const CLASS_MAP = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9
};

const CLASS_NAMES = Object.keys(CLASS_MAP);


// ----- Mutable globals -----
let embeddings       = [];
let actualLabels     = [];
let predictedLabels  = [];
let filenames        = [];
let cues             = {};
let isLabeled        = [];
let datasetIndices   = [];
let trueCodes        = [];

// all the derived arrays we need
let xAll = [], yAll = [], zAll = [];
let uncertainty = [], density = [], diversity = [], novelty = [], coverage = [];
let uncertaintyNorm = [], diversityNorm = [], noveltyNorm = [], densityNorm = [], coverageNorm = [];
let diversityBins = [];
const opacityPerBin = [0.15, 0.45, 0.9];  // low / medium / high
let hoverTextsAll = [];
let classToIndex = {};
let colorIndexAll = [];
let nClasses = 0;
let colorScale = [];
let sizeAll = [];
let symbolAll = [];
let UNC_HIGH = NaN, DIV_HIGH = NaN, NOV_HIGH = NaN;

// ----- Helpers -----
function normalize(arr) {
    if (!arr.length) return arr;
    const finite = arr.filter(v => Number.isFinite(v));
    if (!finite.length) return arr;
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    const range = max - min || 1.0;
    return arr.map(v => (v - min) / range);
}

function quantile(normArr, q) {
    const finite = normArr.filter(Number.isFinite);
    if (!finite.length) return NaN;
    const sorted = [...finite].sort((a, b) => a - b);
    const idx = Math.floor(q * (sorted.length - 1));
    return sorted[idx];
}

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

function buildClassLegend() {
    const legendDiv = document.getElementById("classLegend");
    if (!legendDiv) return;

    let html = "<strong>Class colors (predicted):</strong><br>";
    const entries = Object.entries(classToIndex).sort((a, b) => a[1] - b[1]);

    const classColors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];

    entries.forEach(([lbl, idx]) => {
        const color = classColors[idx % classColors.length];
        html += `<span style="display:inline-block;width:12px;height:12px;background:${color};border:1px solid #000;margin-right:4px;"></span>`;
        html += `<span>${lbl}</span><br>`;
    });
    legendDiv.innerHTML = html;
}

// ----- Apply data snapshot (initial + later refreshes) -----
function applyEmbeddingData(data) {
    // 1) Raw data
    embeddings      = data.embeddings || [];
    actualLabels    = data.actual_labels || [];
    predictedLabels = data.predicted_labels || [];
    filenames       = data.filenames || [];
    cues            = data.cues || {};
    const isLabeledRaw = data.is_labeled || [];
    datasetIndices  = data.indices || [];
    trueCodes       = data.true_codes || [];

    isLabeled = isLabeledRaw.map(v => (v === true || v === "true" || v === 1));

    // 2) Basic coords
    xAll = embeddings.map(e => e[0]);
    yAll = embeddings.map(e => e[1]);
    zAll = embeddings.map(e => e[2]);

    // 3) Cues
    uncertainty = cues["uncertainty"] || [];
    density     = cues["density"]     || [];
    diversity   = cues["diversity"]   || [];
    novelty     = cues["novelty"]     || [];
    coverage    = cues["coverage"]    || [];

    // 4) Normalization
    uncertaintyNorm = normalize(uncertainty);
    diversityNorm   = normalize(diversity);
    noveltyNorm     = normalize(novelty);
    densityNorm     = normalize(density);
    coverageNorm    = normalize(coverage);

    // 5) Thresholds (recomputed each time)
    UNC_HIGH = quantile(uncertaintyNorm, 0.66);
    DIV_HIGH = quantile(diversityNorm,   0.66);
    NOV_HIGH = quantile(noveltyNorm,     0.66);

    // 6) Diversity bins
    diversityBins = binDiversity(diversityNorm);

    // 7) Hover texts
    hoverTextsAll = filenames.map((f, i) =>
        `<b>${f}</b>` +
        //`<br>Actual: ${actualLabels[i]}` +
        `<br>Predicted: ${predictedLabels[i]}` +
        `<br>is_labeled: ${isLabeled[i]}` +
        `<br>uncertainty: ${uncertainty[i]?.toFixed(3)}` +
        `<br>density: ${density[i]?.toFixed(3)}` +
        `<br>diversity: ${diversity[i]?.toFixed(3)}` +
        `<br>novelty: ${novelty[i]?.toFixed(3)}` +
        `<br>coverage: ${coverage[i]?.toFixed(3)}`
    );

    // 8) Class color mapping
    classToIndex = {};
    let classCounter = 0;
    predictedLabels.forEach(lbl => {
        if (!(lbl in classToIndex)) {
            classToIndex[lbl] = classCounter++;
        }
    });
    colorIndexAll = predictedLabels.map(lbl => classToIndex[lbl]);

    const classColors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];

    nClasses = Object.keys(classToIndex).length;
    colorScale = [];
    for (let i = 0; i < nClasses; i++) {
        const t = (nClasses === 1) ? 0.5 : i / (nClasses - 1);
        const color = classColors[i % classColors.length];
        colorScale.push([t, color]);
    }

    // 9) Marker size & symbol
    sizeAll = uncertaintyNorm.map(u => 3 + 17 * (u || 0));  // 3..20

    symbolAll = (function() {
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

    // 10) Rebuild legend & redraw plot
    buildClassLegend();
    updatePlot();
}

// expose to other scripts (metrics.js)
window.applyEmbeddingData = applyEmbeddingData;

// ----- Plot + interactions -----
const hideCheckbox        = document.getElementById("hideLabeledCheckbox");
const autoRotateCheckbox  = document.getElementById("autoRotateCheckbox");
const gd                  = document.getElementById('plot');
const hoverInfoEl         = document.getElementById('hoverInfo');
const cueSelect    = document.getElementById("cueSelect");
const cueSlider    = document.getElementById("cueSlider");
const cuePctLabel  = document.getElementById("cuePctLabel");
const K_FIXED = 20;


cueSelect.addEventListener("change", updatePlot);

cueSlider.addEventListener("input", () => {
    cuePctLabel.textContent = Math.round(cueSlider.value * 100);
    updatePlot();
});

if (cuePctLabel) {
    cuePctLabel.textContent = Math.round(cueSlider.value * 100);
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

let angle = 0;
let isSpinning = false;
let spinFrameId = null;
let eventsAttached = false;

function promptForLabel() {
    const options = CLASS_NAMES
        .map((c, i) => `${i}: ${c}`)
        .join("\n");

    const input = prompt(
        "Select label by number:\n" + options
    );

    if (input === null) return null;

    const idx = parseInt(input);
    if (isNaN(idx) || idx < 0 || idx >= CLASS_NAMES.length) {
        alert("Invalid selection");
        return null;
    }

    return CLASS_NAMES[idx];
}


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

function makeData(hideLabeledFlag) {
    const baseVisibleIdxs = getBaseVisibleIndices(hideLabeledFlag);

    const covIdxGlobal = computeHotspotIndices(K_FIXED, coverageNorm, baseVisibleIdxs);
    const denIdxGlobal = computeHotspotIndices(K_FIXED, densityNorm, baseVisibleIdxs);
    const covSet = new Set(covIdxGlobal);
    const denSet = new Set(denIdxGlobal);
    let visibleIdxs = baseVisibleIdxs.slice();

    const cueName = cueSelect?.value || "none";
    const q       = parseFloat(cueSlider?.value ?? 1);

    visibleIdxs = filterByCueQuantile(visibleIdxs, cueName, q);


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
                        width: lineWidths,
                        opacity: 1.0
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

function filterByCueQuantile(idxs, cueName, q) {
    if (cueName === "none" || q >= 1) return idxs;

    const cueMap = {
        uncertainty: uncertaintyNorm,
        diversity:   diversityNorm,
        novelty:    noveltyNorm,
        density:    densityNorm,
        coverage:   coverageNorm
    };

    const arr = cueMap[cueName];
    if (!arr || !arr.length) return idxs;

    const values = idxs
        .map(i => arr[i])
        .filter(v => Number.isFinite(v))
        .sort((a, b) => a - b);

    if (!values.length) return idxs;

    // keep TOP q fraction
    const cutoff = values[
        Math.floor((1 - q) * (values.length - 1))
    ];

    return idxs.filter(i =>
        Number.isFinite(arr[i]) && arr[i] >= cutoff
    );
}


function attachPlotEvents() {
    if (eventsAttached) return;
    eventsAttached = true;

    // Hover feedback
    gd.on('plotly_hover', evt => {
        const pt = evt.points[0];
        const globalIdx = pt.customdata;
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
    gd.on('plotly_click', async (evt) => {
        if (window.isRetraining) {
            alert("The model is currently retraining. Please wait.");
            return;
        }

        const pt = evt.points[0];
        const globalIdx = pt.customdata;
        const filename = filenames[globalIdx];
        const dsIdx = datasetIndices[globalIdx];
        const predStr = predictedLabels[globalIdx];

        try {
            // Play audio
            const audio = new Audio(`/audio/${filename}`);
            await audio.play();


            const selectedClass = promptForLabel();
            if (!selectedClass) return;

            const classCode = CLASS_MAP[selectedClass];

            if (!window.lastUserName) {
                const name = prompt("Annotator name:", "user1");
                if (!name) return;
                window.lastUserName = name;
            }
            const res = await fetch("/human_annotations", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    filenames: [filename],
                    indices: [dsIdx],
                    labels: [classCode],
                    user: window.lastUserName
                })
            });

            const data = await res.json();
            console.log("Annotation stored:", data);
            hoverInfoEl.innerHTML =
                `Annotated: <strong>${filename}</strong>` +
                ` · label=${selectedClass}` +
                ` · code=${classCode}` +
                ` · user=${window.lastUserName}`;
        } catch (err) {
            console.error("Audio failed to play", err);
            alert(`Failed to play audio for: ${filename}`);
        }
    });

}

function updatePlot() {
    if (!gd) return;
    const hideFlag   = hideCheckbox.checked;
    const newData = makeData(hideFlag);
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

autoRotateCheckbox.addEventListener('change', () => {
    if (autoRotateCheckbox.checked) {
        startSpin();
    } else {
        stopSpin();
    }
});



// Initial render using server-injected data
if (window.IMLVR_DATA) {
    applyEmbeddingData(window.IMLVR_DATA);
}
