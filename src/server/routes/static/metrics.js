// ----- METRICS PANEL -----
const globalAccTextEl = document.getElementById("globalAccText");
const globalAccSubEl  = document.getElementById("globalAccSub");
const globalAccBarEl  = document.getElementById("globalAccBar");
const perClassListEl  = document.getElementById("perClassList");

function renderMetrics(m) {
    const acc           = m.accuracy;
    const target        = m.accuracy_target || 1.0;
    const totalLabeled  = m.total_labeled ?? null;
    const humanLabeled  = m.human_labeled ?? null;
    const perClassAcc   = m.per_class_accuracy || {};
    const labeledCounts = m.labeled_counts || {};
    const phase         = m.phase || null;

    if (typeof acc === "number" && !isNaN(acc)) {
        const accPct = (acc * 100).toFixed(1);
        globalAccTextEl.textContent = `Accuracy: ${accPct}%`;

        const safeTarget = target > 0 ? target : 1.0;
        const ratio = Math.max(0, Math.min(1, acc / safeTarget));
        globalAccBarEl.style.width = (ratio * 100).toFixed(1) + "%";

        // ✨ phase-aware subtitle
        if (phase === "retrain_start") {
            globalAccSubEl.textContent =
                `Retraining on ${humanLabeled ?? "?"} human labels…`;
        } else {
            let sub = `Target: ${(safeTarget * 100).toFixed(1)}%`;
            if (totalLabeled != null) {
                sub += ` · Labeled: ${totalLabeled}`;
            }
            if (humanLabeled != null) {
                sub += ` (human: ${humanLabeled})`;
            }
            globalAccSubEl.textContent = sub;
        }
    } else {
        globalAccTextEl.textContent = "Accuracy: –";
        globalAccSubEl.textContent = "Waiting for metrics…";
        globalAccBarEl.style.width = "0%";
    }

    // Per-class list
    const entries = Object.entries(perClassAcc).sort((a, b) => a[0].localeCompare(b[0]));
    if (!entries.length) {
        perClassListEl.innerHTML = "<em>No per-class metrics yet.</em>";
        return;
    }

    let html = "";
    for (const [cls, accC] of entries) {
        const accPct = (accC * 100).toFixed(1);
        const count = labeledCounts[cls] || 0;
        const discovered = count > 0;

        const barWidth = Math.max(3, Math.min(100, accC * 100)); // 3..100%
        const barColor = discovered ? "#3b82f6" : "#9ca3af";

        const badgeText  = discovered ? `${count} labeled` : "0 labeled · not discovered";
        const badgeColor = discovered ? "#dcfce7" : "#f3f4f6";
        const badgeBorder= discovered ? "#22c55e" : "#d4d4d8";
        const badgeTextColor = discovered ? "#166534" : "#4b5563";

        html += `
            <div style="display:flex; align-items:center; margin-bottom:0.25rem;">
                <div style="flex:0 0 90px; font-weight:500;">${cls}</div>
                <div style="flex:1; margin:0 0.75rem;">
                    <div style="width:100%; height:8px; border-radius:999px; background:#e5e7eb; overflow:hidden;">
                        <div style="
                            height:100%;
                            width:${barWidth}%;
                            background:${barColor};
                            transition:width 0.4s ease;
                        "></div>
                    </div>
                </div>
                <div style="flex:0 0 60px; font-variant-numeric:tabular-nums; text-align:right; margin-right:0.5rem;">
                    ${accPct}%
                </div>
                <div style="
                    flex:0 0 auto;
                    font-size:0.7rem;
                    padding:0.1rem 0.45rem;
                    border-radius:999px;
                    border:1px solid ${badgeBorder};
                    background:${badgeColor};
                    color:${badgeTextColor};
                ">
                    ${badgeText}
                </div>
            </div>
        `;
    }
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
// And once at startup
pollMetrics();
