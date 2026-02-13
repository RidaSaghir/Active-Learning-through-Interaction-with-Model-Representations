console.log("[BOOT] embeddings.js loaded");

const cueThresholds = {
  uncertainty: 0,
  diversity: 0,
  novelty: 0,
  density: 0,
  coverage: 0
};
window.cueThresholds = cueThresholds;

// Cached percentile cutoffs (REUSED during filtering)
let cueCutoffs = {
  uncertainty: -Infinity,
  diversity: -Infinity,
  novelty: -Infinity,
  density: -Infinity,
  coverage: -Infinity
};

// Bind once (avoid object creation in loops)
let cueArrays = {};



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
const classColors = [
    '#1f77b4', // air_conditioner
    '#ff7f0e', // car_horn
    '#2ca02c', // children_playing
    '#d62728', // dog_bark
    '#9467bd', // drilling
    '#8c564b', // engine_idling
    '#e377c2', // gun_shot
    '#7f7f7f', // jackhammer
    '#bcbd22', // siren
    '#17becf'  // street_music
];

const CLASS_NAMES = Object.keys(CLASS_MAP);

const VISUAL_CONFIG = {
    color: true,        // predicted class
    size: false,        // uncertainty
    opacity: false,     // diversity
    shape: false,       // novelty
    border: false       // class coverage / density
};

// ----- Mutable globals -----
let embeddings       = [];
let actualLabels     = [];
let predictedLabels  = [];
let filenames        = [];
let cues             = {};
let isLabeled        = [];
let datasetIndices   = [];
let trueCodes        = [];
let audioPlayer = new Audio();
audioPlayer.preload = "auto";
let globalXRange = null;
let globalYRange = null;
let globalZRange = null;
let eventsAttached = false;
let isAudioLoading = false;


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
    // Determine embedding dimensionality (ONCE per update)
    const embeddingDim =
        data.embedding_dim ??
        (data.embeddings?.[0]?.length ?? 3);

    window.EMBEDDING_DIM = embeddingDim;
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
// 2) Basic coords
    xAll = embeddings.map(e => e[0]);
    yAll = embeddings.map(e => e[1]);
    zAll = (window.EMBEDDING_DIM === 3)
        ? embeddings.map(e => e[2])
        : embeddings.map(_ => 0);  // dummy z for 2D

    // 🔒 Fix global ranges once so filtering doesn't change them
    if (xAll.length) {
      const xMin = Math.min(...xAll), xMax = Math.max(...xAll);
      const yMin = Math.min(...yAll), yMax = Math.max(...yAll);
      const zMin = Math.min(...zAll), zMax = Math.max(...zAll);

      // optional small padding
      const pad = 0.05;
      globalXRange = [
        xMin - pad * (xMax - xMin),
        xMax + pad * (xMax - xMin)
      ];
      globalYRange = [
        yMin - pad * (yMax - yMin),
        yMax + pad * (yMax - yMin)
      ];
      globalZRange = [
        zMin - pad * (zMax - zMin || 1),
        zMax + pad * (zMax - zMin || 1)
      ];
    }


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
        `<br>is_labeled: ${isLabeled[i]}`
        // `<br>uncertainty: ${uncertainty[i]?.toFixed(3)}` +
        // `<br>density: ${density[i]?.toFixed(3)}` +
        // `<br>diversity: ${diversity[i]?.toFixed(3)}` +
        // `<br>novelty: ${novelty[i]?.toFixed(3)}` +
        // `<br>coverage: ${coverage[i]?.toFixed(3)}`
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

    nClasses = Object.keys(classToIndex).length;
    colorScale = [];
    for (let i = 0; i < nClasses; i++) {
        const t0 = i / nClasses;
        const t1 = (i + 1) / nClasses;
        colorScale.push([t0, classColors[i]]);
        colorScale.push([t1, classColors[i]]);
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

    // ebuild legend & redraw plot
    //buildClassLegend();
    updatePlot();
    cueArrays = {
      uncertainty: uncertaintyNorm,
      diversity: diversityNorm,
      novelty: noveltyNorm,
      density: densityNorm,
      coverage: coverageNorm
    };

    // Initialize cutoffs once
    recomputeCueCutoffs();

}

// expose to other scripts (metrics.js)
window.applyEmbeddingData = applyEmbeddingData;

// ----- Plot + interactions -----
const hideCheckbox        = document.getElementById("hideLabeledCheckbox");
const gd                  = document.getElementById('plot');
console.log("[BOOT] gd id:", gd?.id, "node:", gd);
const hoverInfoEl         = document.getElementById('hoverInfo');
const K_FIXED = 20;


let angle = 0;
let isSpinning = false;
let spinFrameId = null;

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
    const PLOT_TYPE =
        (window.EMBEDDING_DIM === 3) ? 'scatter3d' : 'scatter';
    const baseVisibleIdxs = getBaseVisibleIndices(hideLabeledFlag);

    const covIdxGlobal = computeHotspotIndices(K_FIXED, coverageNorm, baseVisibleIdxs);
    const denIdxGlobal = computeHotspotIndices(K_FIXED, densityNorm, baseVisibleIdxs);
    const covSet = new Set(covIdxGlobal);
    const denSet = new Set(denIdxGlobal);
    let visibleIdxs = baseVisibleIdxs.filter(i => passesAllActiveCues(i));

    if (!visibleIdxs.length) {
        console.warn("No samples pass active cue filters", cueThresholds);
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

            const trace = {
                name: `${cls} (div bin ${bin})`,
                x: pick(xAll, idxsThis),
                y: pick(yAll, idxsThis),
                mode: 'markers',
                type: PLOT_TYPE,
                text: pick(hoverTextsAll, idxsThis),
                hoverinfo: 'text',
                customdata: originalIndices,
                marker: {
                    size: VISUAL_CONFIG.size
                        ? pick(sizeAll, idxsThis)
                        : 8,

                    color: pick(colorIndexAll, idxsThis),
                    colorscale: colorScale,
                    cmin: 0,
                    cmax: nClasses,
                    opacity: VISUAL_CONFIG.opacity
                        ? opacityPerBin[bin]
                        : 0.9,
                    symbol: VISUAL_CONFIG.symbol
                        ? pick(symbolAll, idxsThis)
                        : "circle",
                     line: VISUAL_CONFIG.border
                        ? {
                        color: lineColors,
                        width: lineWidths,
                        opacity: 1.0
                        }
                        : {
                            width: 0                  // no borders
                        },
                    colorbar: {
                        title: 'Predicted class',
                     //   thickness: 15
                        tickmode: "array",
                        tickvals: [...Array(nClasses).keys()].map(i => i + 0.5),
                        ticktext: Object.keys(classToIndex),
                        len: 0.6
                    }
                },
                showscale: (clsIdx === 0 && bin === 0),
                //showlegend: false
            };
            if (window.EMBEDDING_DIM === 3) {
                trace.z = pick(zAll, idxsThis);
            }
            traces.push(trace);


        }
    }

    return traces;
}



function attachPlotEvents() {
    if (!gd) {
        console.warn("[PLOT] gd is null - cannot attach events");
        return;
    }
    if (eventsAttached) return;
    eventsAttached = true;

    console.log("[PLOT] attachPlotEvents called");
    // OPTIONAL: show hover info below plot (Plotly hover still works)
    gd.on('plotly_hover', (evt) => {
        console.log("[HOVER] Hover event", evt)
        if (!evt.points || evt.points.length === 0) return;
        const pt = evt.points[0];
        const i = pt.customdata;
        if (i == null) return;

        hoverInfoEl.innerHTML =
            `Selected: <strong>${filenames[i]}</strong>` +
            ` · Pred: ${predictedLabels[i]}` ;
            // ` · Unc: ${uncertainty[i]?.toFixed(2)}` +
            // ` · Div: ${diversity[i]?.toFixed(2)}` +
            // ` · Nov: ${novelty[i]?.toFixed(2)}`;
    });

    gd.on('plotly_unhover', () => {
        hoverInfoEl.innerHTML = "";
    });


    //CLICK → play audio + label
    gd.on('plotly_click', async (evt) => {
        console.log("[CLICK] event fired", evt)
        if (window.isRetraining || isAudioLoading) {
            console.log("[CLICK] blocked: retraining")
            alert("The model is currently retraining or an audio is processing. Please wait.");
            return;
        }

        if (!evt.points || evt.points.length === 0) {
            console.log("[CLICK] no point selected (background click)");
            return;
            }

        console.log("[CLICK] points length =", evt.points.length);

        const globalIdx = evt.points[0].customdata;
        console.log("[CLICK] globalIdx =", globalIdx)
        if (globalIdx == null) return;

        const filename = filenames[globalIdx];
        isAudioLoading = true
        console.log("[CLICK] filename =", filename);
        if (!filename) return;

        try {
            audioPlayer.pause();
            //audioPlayer.currentTime = 0;
            audioPlayer.src = `/audio/${filename}?t=${Date.now()}`;
            audioPlayer.load()
            console.log("[AUDIO] src =", audioPlayer.src);

            await audioPlayer.play();
            console.log("[AUDIO] play() resolved");

            setTimeout(async () => {
                const selectedClass = promptForLabel();
                if (selectedClass) {
                    const datasetIdx = datasetIndices[globalIdx];
                    const labelCode = CLASS_MAP[selectedClass];

                    await fetch("/human_annotations", {
                        method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            filenames: [filename],
                            indices: [datasetIdx],
                            labels: [labelCode],
                            user: "rida"
                          })
                        });
                                    // Handle your labeling logic here (e.g., send to server)
                    console.log(`Labeled ${filename} as ${selectedClass}`);
                }
                isAudioLoading = false;
            }, 100);

        } catch (err) {
            console.error("Audio playback failed:", err);
            alert(`Failed to play audio for: ${filename}`);
            isAudioLoading = false
        }

    });
    console.log("[PLOT] plotly_click/hover handlers attached");
}


function updatePlot() {
    console.log("[PLOT] updatePlot called");
    if (!gd) return;
    const hideFlag   = hideCheckbox.checked;
    const newData = makeData(hideFlag);
    Plotly.react(gd, newData, getLayout(), {
        displaylogo: false,
        modeBarButtonsToRemove: [
            'toImage', 'select2d', 'lasso2d',
            'hoverCompareCartesian', 'hoverClosestCartesian',
             // 3D camera/nav buttons (removes rotate/pan/zoom UI)
            'orbitRotation',
            'tableRotation',
            'pan3d',
            'zoom3d',
            'resetCameraDefault3d',
            'resetCameraLastSave3d',
            'hoverClosest3d'
        ]
    }).then(() => {
        attachPlotEvents();

    });
}

function getLayout() {
    if (window.EMBEDDING_DIM === 3) {
        return {
            scene: {
                aspectmode: "cube",
                camera: { eye: { x: 1.8, y: 1.8, z: 1.4 } },
                xaxis: globalXRange ? { range: globalXRange } : {},
                yaxis: globalYRange ? { range: globalYRange } : {},
                zaxis: globalZRange ? { range: globalZRange } : {}
            },
            dragmode: false,
            margin: {l: 0, r: 0, t: 0, b: 0},
            showlegend: false
        };
    } else {
        return {
            hovermode: "closest",
            dragmode: false,
            clickmode: "event",
            hoverdistance:1,
            spikedistance:-1,
            xaxis: {
                zeroline: false,
                showspikes: false,
                range: globalXRange || undefined
            },
            yaxis: {
                zeroline: false,
                showspikes: false,
                range: globalYRange || undefined
            },
            margin: {l: 0, r: 0, t: 0, b: 0},
            showlegend: false
        };

    }
}

function buildMarker(idxsThis, bin) {
    const marker = {
        color: pick(colorIndexAll, idxsThis),
        colorscale: colorScale,
        cmin: 0,
        cmax: nClasses - 1
    };

    if (VISUAL_CONFIG.size) {
        marker.size = pick(sizeAll, idxsThis);
    } else {
        marker.size = 8; // fixed size
    }

    if (VISUAL_CONFIG.opacity) {
        marker.opacity = opacityPerBin[bin];
    } else {
        marker.opacity = 0.9;
    }

    if (VISUAL_CONFIG.shape) {
        marker.symbol = pick(symbolAll, idxsThis);
    } else {
        marker.symbol = "circle";
    }

    if (VISUAL_CONFIG.border) {
        marker.line = {
            color: pickBorderColors(idxsThis),
            width: pickBorderWidths(idxsThis),
            opacity: 1.0
        };
    }

    return marker;
}

function pickBorderColors(idxs) {
    return idxs.map(i =>
        coverageNorm[i] > UNC_HIGH ? "red" :
        densityNorm[i] > UNC_HIGH  ? "black" :
        "rgba(0,0,0,0)"
    );
}

function pickBorderWidths(idxs) {
    return idxs.map(i =>
        coverageNorm[i] > UNC_HIGH || densityNorm[i] > UNC_HIGH ? 2 : 0
    );
}

function percentileCutoff(normArr, keepFrac) {

  const finite = normArr.filter(Number.isFinite);
  if (!finite.length) return Infinity;
  if (keepFrac >= 1) {
    // 100% → no filtering (cutoff below all values)
    return -Infinity;
  }

  const sorted = [...finite].sort((a, b) => a - b);

  // keep top keepFrac → cutoff at (1 - keepFrac)
  const idx = Math.floor((1 - keepFrac) * (sorted.length - 1));
  return sorted[idx];
}


function passesAllActiveCues(i) {
  // Only apply a cue if its slider > 0

  if (cueThresholds.uncertainty > 0 &&
      uncertaintyNorm[i] < cueCutoffs.uncertainty) return false;

  if (cueThresholds.diversity > 0 &&
      diversityNorm[i] < cueCutoffs.diversity) return false;

  if (cueThresholds.novelty > 0 &&
      noveltyNorm[i] < cueCutoffs.novelty) return false;

  if (cueThresholds.density > 0 &&
      densityNorm[i] < cueCutoffs.density) return false;

  if (cueThresholds.coverage > 0 &&
      coverageNorm[i] < cueCutoffs.coverage) return false;

  return true;
}


function recomputeCueCutoffs() {
  // 0 → cue off → cutoff so low that nothing is filtered by that cue
  cueCutoffs.uncertainty = (cueThresholds.uncertainty > 0)
    ? percentileCutoff(uncertaintyNorm, cueThresholds.uncertainty)
    : -Infinity;

  cueCutoffs.diversity = (cueThresholds.diversity > 0)
    ? percentileCutoff(diversityNorm, cueThresholds.diversity)
    : -Infinity;

  cueCutoffs.novelty = (cueThresholds.novelty > 0)
    ? percentileCutoff(noveltyNorm, cueThresholds.novelty)
    : -Infinity;

  cueCutoffs.density = (cueThresholds.density > 0)
    ? percentileCutoff(densityNorm, cueThresholds.density)
    : -Infinity;

  cueCutoffs.coverage = (cueThresholds.coverage > 0)
    ? percentileCutoff(coverageNorm, cueThresholds.coverage)
    : -Infinity;
}


// Controls
hideCheckbox.addEventListener('change', updatePlot);

let plotRAF = null;

function requestPlotUpdate() {
  if (plotRAF !== null) return;

  plotRAF = requestAnimationFrame(() => {
    plotRAF = null;
    updatePlot();
  });
}

// Initial render using server-injected data
if (window.IMLVR_DATA) {
    applyEmbeddingData(window.IMLVR_DATA);
}

document.querySelectorAll(".cue-control").forEach(ctrl => {
  const cue = ctrl.dataset.cue;
  const slider = ctrl.querySelector("input");

    slider.addEventListener("input", () => {
      const value = parseFloat(slider.value);

      // Update thresholds map
      cueThresholds[cue] = value;

      // Update label
      const valSpan = document.querySelector(`.cue-value[data-cue="${cue}"]`);
      if (valSpan) valSpan.textContent = `Top ${Math.round(value * 100)}%`;
      console.log("LABEL SET TO:", `Top ${Math.round(value * 100)}%`);


      //  Recompute and redraw
      recomputeCueCutoffs();
      requestPlotUpdate();
    });

});
