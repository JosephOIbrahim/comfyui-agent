/* ── Experience Dashboard ───────────────────────────────────────── */

export function createExperienceDash(container, client) {
  const el = document.createElement("div");
  el.className = "sdp-experience";
  container.appendChild(el);

  async function refresh() {
    try {
      const data = await client.getExperience();
      render(data);
    } catch (e) {
      el.innerHTML = `<div style="color:var(--cz-text-muted);padding:var(--cz-4)">Could not load experience data</div>`;
    }
  }

  function render(data) {
    el.innerHTML = "";

    // Learning Phase
    _section(el, "Learning Phase");
    const phaseLabel = document.createElement("div");
    phaseLabel.className = "sdp-phase-label";
    const phaseNames = { prior: "Phase 1 — Prior", blended: "Phase 2 — Blended", experienced: "Phase 3 — Experienced" };
    phaseLabel.textContent = `${phaseNames[data.learning_phase] || data.learning_phase} (${data.total_generations || 0} generations)`;
    el.appendChild(phaseLabel);

    const thresholds = { prior: 30, blended: 100, experienced: 500 };
    const target = thresholds[data.learning_phase] || 100;
    const pct = Math.min(100, ((data.total_generations || 0) / target) * 100);

    const bar = document.createElement("div");
    bar.className = "sdp-phase-bar";
    bar.innerHTML = `<div class="sdp-phase-bar__fill" style="width:${pct}%"></div>`;
    el.appendChild(bar);

    const count = document.createElement("div");
    count.className = "sdp-phase-count";
    count.textContent = `${data.total_generations || 0}/${target}`;
    if (data.learning_phase !== "experienced") {
      const nextPhase = data.learning_phase === "prior" ? "Phase 2" : "Phase 3";
      count.textContent += ` \u2192 ${nextPhase}`;
    }
    el.appendChild(count);

    // Stats
    if (data.avg_quality !== undefined || data.best_quality !== undefined) {
      const statsSection = document.createElement("div");
      statsSection.style.marginTop = "var(--cz-4)";
      _section(statsSection, "Quality Stats");

      const statsLine = document.createElement("div");
      statsLine.className = "sdp-stats-line";
      statsLine.innerHTML = `
        <span>Avg: <span class="sdp-stats-line__val">${(data.avg_quality || 0).toFixed(2)}</span></span>
        <span class="sdp-stats-line__sep">\u00b7</span>
        <span>Best: <span class="sdp-stats-line__val">${(data.best_quality || 0).toFixed(2)}</span></span>
        <span class="sdp-stats-line__sep">\u00b7</span>
        <span>Weight: <span class="sdp-stats-line__val">${(data.experience_weight || 0).toFixed(2)}</span></span>
      `;
      statsSection.appendChild(statsLine);
      el.appendChild(statsSection);
    }

    // Generation counts
    if (data.successful !== undefined) {
      const countsSection = document.createElement("div");
      countsSection.style.marginTop = "var(--cz-4)";
      _section(countsSection, "Generations");

      const countsLine = document.createElement("div");
      countsLine.className = "sdp-stats-line";
      countsLine.innerHTML = `
        <span>Successful: <span class="sdp-stats-line__val">${data.successful || 0}</span></span>
        <span class="sdp-stats-line__sep">\u00b7</span>
        <span>Failed: <span class="sdp-stats-line__val">${data.failed || 0}</span></span>
      `;
      countsSection.appendChild(countsLine);
      el.appendChild(countsSection);
    }

    // Prediction accuracy chart placeholder
    if (data.accuracy_history && data.accuracy_history.length > 0) {
      const chartSection = document.createElement("div");
      chartSection.style.marginTop = "var(--cz-4)";
      _section(chartSection, "Prediction Accuracy");
      chartSection.appendChild(_drawChart(data.accuracy_history));
      el.appendChild(chartSection);
    }

    // Message if empty
    if (!data.total_generations) {
      const msg = document.createElement("div");
      msg.style.cssText = "color:var(--cz-text-muted);font-size:var(--cz-body-sm);margin-top:var(--cz-4)";
      msg.textContent = data.message || "No experience data yet. Generate some images to start learning.";
      el.appendChild(msg);
    }
  }

  return { refresh };
}

function _section(parent, label) {
  const lbl = document.createElement("div");
  lbl.className = "sdp-section__label";
  lbl.textContent = label;
  parent.appendChild(lbl);

  const rule = document.createElement("div");
  rule.className = "sdp-section__rule";
  parent.appendChild(rule);
}

function _drawChart(values) {
  const canvas = document.createElement("canvas");
  canvas.className = "sdp-chart";
  canvas.width = 300;
  canvas.height = 60;

  // Draw after append (needs dimensions)
  requestAnimationFrame(() => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = 4;
    const n = values.length;
    if (n < 2) return;

    const stepX = (w - pad * 2) / (n - 1);

    ctx.clearRect(0, 0, w, h);

    // Line
    ctx.beginPath();
    ctx.strokeStyle = "#2A2A2A";
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const x = pad + i * stepX;
      const y = h - pad - (values[i] * (h - pad * 2));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dots
    for (let i = 0; i < n; i++) {
      const x = pad + i * stepX;
      const y = h - pad - (values[i] * (h - pad * 2));
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = "#0066FF";
      ctx.fill();
    }
  });

  return canvas;
}
