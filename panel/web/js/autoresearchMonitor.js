/* ── Autoresearch Monitor — Overnight Results ──────────────────── */

export function createAutoresearchMonitor(container, client) {
  const el = document.createElement("div");
  el.className = "sdp-research";
  container.appendChild(el);

  async function refresh() {
    try {
      const data = await client.getAutoresearch();
      render(data);
    } catch (e) {
      el.innerHTML = `<div style="color:var(--cz-text-muted);padding:var(--cz-4)">Could not load autoresearch data</div>`;
    }
  }

  function render(data) {
    el.innerHTML = "";

    if (data.status === "idle" || !data.title) {
      const msg = document.createElement("div");
      msg.style.cssText = "color:var(--cz-text-muted);font-size:var(--cz-body-sm)";
      msg.textContent = data.message || "No autoresearch run active. Start one from the chat with a prompt like: \"optimize this portrait workflow overnight\"";
      el.appendChild(msg);
      return;
    }

    // Title
    const title = document.createElement("div");
    title.className = "sdp-research__title";
    title.textContent = `AUTORESEARCH: "${data.title}"`;
    el.appendChild(title);

    // Meta line
    const meta = document.createElement("div");
    meta.className = "sdp-research__meta";

    const statusClass = {
      complete: "sdp-status--complete",
      running: "sdp-status--running",
      failed: "sdp-status--failed",
    }[data.status] || "";

    const duration = data.duration_ms ? _formatDuration(data.duration_ms) : "";
    meta.innerHTML = `
      <span class="${statusClass}">${_esc(data.status.toUpperCase())}</span>
      <span>\u00b7</span>
      <span>${data.iterations || 0} iterations</span>
      ${duration ? `<span>\u00b7</span><span>${duration}</span>` : ""}
      ${data.ratchet_intact ? `<span>\u00b7</span><span>Ratchet: \u2713</span>` : ""}
    `;
    el.appendChild(meta);

    // Quality trajectory chart
    if (data.quality_trajectory && data.quality_trajectory.length > 0) {
      const chartSection = document.createElement("div");
      chartSection.style.marginTop = "var(--cz-4)";
      _sectionLabel(chartSection, "Quality Trajectory");
      chartSection.appendChild(_drawTrajectory(data.quality_trajectory));
      el.appendChild(chartSection);
    }

    // Winning parameters
    if (data.winning_params && Object.keys(data.winning_params).length > 0) {
      const paramsSection = document.createElement("div");
      paramsSection.style.marginTop = "var(--cz-4)";
      _sectionLabel(paramsSection, "Winning Parameters");

      for (const [param, info] of Object.entries(data.winning_params)) {
        const row = document.createElement("div");
        row.className = "sdp-param-change";

        const name = document.createElement("span");
        name.className = "sdp-param-change__name";
        name.textContent = param;
        row.appendChild(name);

        const values = document.createElement("span");
        values.className = "sdp-param-change__values";
        values.textContent = `${info.before} \u2192 ${info.after}`;
        row.appendChild(values);

        if (info.note) {
          const note = document.createElement("span");
          note.className = "sdp-param-change__note";
          note.textContent = info.note;
          row.appendChild(note);
        }

        paramsSection.appendChild(row);
      }
      el.appendChild(paramsSection);
    }

    // Summary
    if (data.variants_tried !== undefined) {
      const summary = document.createElement("div");
      summary.className = "sdp-stats-line";
      summary.style.marginTop = "var(--cz-4)";
      summary.innerHTML = `
        <span>Tried: <span class="sdp-stats-line__val">${data.variants_tried}</span></span>
        <span class="sdp-stats-line__sep">\u00b7</span>
        <span>Kept: <span class="sdp-stats-line__val">${data.kept || 0}</span></span>
        <span class="sdp-stats-line__sep">\u00b7</span>
        <span>Discarded: <span class="sdp-stats-line__val">${data.discarded || 0}</span></span>
      `;
      el.appendChild(summary);
    }

    // Actions
    const actions = document.createElement("div");
    actions.className = "sdp-research__actions";
    actions.style.marginTop = "var(--cz-3)";

    if (data.status === "complete") {
      const applyBtn = document.createElement("button");
      applyBtn.className = "sdp-btn sdp-btn--primary";
      applyBtn.textContent = "Apply winning params";
      applyBtn.addEventListener("click", async () => {
        applyBtn.disabled = true;
        applyBtn.textContent = "Applying...";
        try {
          await client.applyPrediction(data.run_id, "winning");
          applyBtn.textContent = "Applied";
        } catch {
          applyBtn.textContent = "Failed";
        }
      });
      actions.appendChild(applyBtn);
    }

    const newBtn = document.createElement("button");
    newBtn.className = "sdp-btn";
    newBtn.textContent = "New run...";
    newBtn.addEventListener("click", () => {
      container.dispatchEvent(new CustomEvent("sdp-command", { detail: { command: "app" } }));
    });
    actions.appendChild(newBtn);

    el.appendChild(actions);
  }

  return { refresh };
}

function _sectionLabel(parent, label) {
  const lbl = document.createElement("div");
  lbl.className = "sdp-section__label";
  lbl.textContent = label;
  parent.appendChild(lbl);
  const rule = document.createElement("div");
  rule.className = "sdp-section__rule";
  parent.appendChild(rule);
}

function _drawTrajectory(values) {
  const canvas = document.createElement("canvas");
  canvas.className = "sdp-trajectory";
  canvas.width = 300;
  canvas.height = 120;

  requestAnimationFrame(() => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = 20;
    const n = values.length;
    if (n < 2) return;

    const stepX = (w - pad * 2) / (n - 1);
    const maxVal = Math.max(...values, 1.0);

    ctx.clearRect(0, 0, w, h);

    // Y-axis labels
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#888888";
    ctx.textAlign = "right";
    for (let v = 0; v <= 1; v += 0.2) {
      const y = h - pad - (v / maxVal) * (h - pad * 2);
      ctx.fillText(v.toFixed(1), pad - 4, y + 4);
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = "#0066FF";
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const x = pad + i * stepX;
      const y = h - pad - (values[i] / maxVal) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dots
    for (let i = 0; i < n; i++) {
      const x = pad + i * stepX;
      const y = h - pad - (values[i] / maxVal) * (h - pad * 2);

      // Check if plateau (ratchet locked)
      const isLocked = i > 0 && values[i] === values[i - 1];
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = isLocked ? "#00CC66" : "#0066FF";
      ctx.fill();
    }

    // X-axis labels (every 5th)
    ctx.fillStyle = "#888888";
    ctx.textAlign = "center";
    for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 6))) {
      const x = pad + i * stepX;
      ctx.fillText(String(i + 1), x, h - 4);
    }
  });

  return canvas;
}

function _formatDuration(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function _esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
