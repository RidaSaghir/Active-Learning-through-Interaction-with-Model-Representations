if (typeof window.isRetraining === "undefined") {
    window.isRetraining = false;
}

const globalAccTextEl = document.getElementById("globalAccText");
const globalAccSubEl  = document.getElementById("globalAccSub");
const globalAccBarEl  = document.getElementById("globalAccBar");
const perClassListEl  = document.getElementById("perClassList");
const iterationLabelEl = document.getElementById("iterationLabel");
const retrainBannerEl   = document.getElementById("retrainBanner");


let lastIterationSeen = window.IMLVR_DATA ? window.IMLVR_DATA.iteration : -1;



function renderMetrics(m) {
    const target        = m.accuracy_target || 1.0;
    const totalLabeled  = m.total_labeled ?? null;
    const humanLabeled  = m.human_labeled ?? null;
    const perClassAcc   = m.per_class_accuracy || {};
    const labeledCounts = m.labeled_counts || {};
    const phase         = m.phase || null;

    if (iterationLabelEl && typeof m.iteration === "number") {
        iterationLabelEl.textContent = `Iteration ${m.iteration}`;
    }

    if (typeof m.iteration === "number" &&
        (lastIterationSeen === null || m.iteration > lastIterationSeen)) {

        lastIterationSeen = m.iteration;

        fetch("/latest_embeddings_json")
            .then(r => r.json())
            .then(data => {
                if (window.applyEmbeddingData && !data.error) {
                    window.applyEmbeddingData(data);
                }
            })
            .catch(err => {
                console.warn("Failed to refresh embeddings:", err);
            });
    }

    //  Special case: retrain start -> keep accuracy, only change subtitle
    if (phase === "retrain_start") {
        window.isRetraining = true;
        globalAccSubEl.textContent =
            `Retraining on ${humanLabeled ?? "?"} human labels…`;
        globalAccSubEl.style.color = "#b45309";  // subtle orange
        if (retrainBannerEl) {
            retrainBannerEl.style.display = "flex";
        }
        window.isRetraining = true;
        return;
    } else {
        globalAccSubEl.style.color = "#6b7280";
        if (retrainBannerEl) {
            retrainBannerEl.style.display = "none";
        }
        window.isRetraining = false;
    }

    const acc = m.accuracy;

    if (typeof acc === "number" && !isNaN(acc)) {
        const accPct = (acc * 100).toFixed(1);
        globalAccTextEl.textContent = `Accuracy: ${accPct}%`;

        const safeTarget = target > 0 ? target : 1.0;
        const ratio = Math.max(0, Math.min(1, acc / safeTarget));
        globalAccBarEl.style.width = (ratio * 100).toFixed(1) + "%";

        let sub = `Target: ${(safeTarget * 100).toFixed(1)}%`;
        if (totalLabeled != null) {
            sub += ` · Labeled: ${totalLabeled}`;
        }
        if (humanLabeled != null) {
            sub += ` (human: ${humanLabeled})`;
        }
        globalAccSubEl.textContent = sub;

    } else {
        globalAccTextEl.textContent = "Accuracy: –";
        globalAccSubEl.textContent = "Waiting for metrics…";
        globalAccBarEl.style.width = "0%";
    }

    // 🔹 Compact per-class view: just text chips, no bars
    const entries = Object.entries(perClassAcc).sort((a, b) => a[0].localeCompare(b[0]));
    if (!entries.length) {
        perClassListEl.innerHTML = "<em>No per-class metrics yet.</em>";
        return;
    }

    let html = `
        <div style="
            display:flex;
            flex-wrap:wrap;
            gap:0.35rem;
        ">
    `;

    for (const [cls, accC] of entries) {
        const accPct = (accC * 100).toFixed(1);
        const count = labeledCounts[cls] || 0;
        const discovered = count > 0;

        const badgeColor     = discovered ? "#eef2ff" : "#f3f4f6";
        const badgeBorder    = discovered ? "#4f46e5" : "#d4d4d8";
        const badgeTextColor = discovered ? "#3730a3" : "#4b5563";
        const suffix         = discovered ? ` · ${count} labeled` : " · 0 labeled";

        html += `
            <span style="
                font-size:0.75rem;
                padding:0.15rem 0.5rem;
                border-radius:999px;
                border:1px solid ${badgeBorder};
                background:${badgeColor};
                color:${badgeTextColor};
                white-space:nowrap;
            ">
                ${cls}: ${accPct}%${suffix}
            </span>
        `;
    }

    html += `</div>`;
    perClassListEl.innerHTML = html;
}


function pollMetrics() {
    fetch("/metrics")
        .then(r => r.json())
        .then(data => {
            if (!data.error) {
                renderMetrics(data);
            }
        })
        .catch(err => {
            console.warn("Failed to fetch metrics:", err);
        });
}

// Poll every 3 seconds
setInterval(pollMetrics, 3000);
pollMetrics();

