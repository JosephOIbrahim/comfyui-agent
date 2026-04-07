/* ── GRAPH Mode — Workflow Inspector ────────────────────────────── */

export function createGraphMode(container, client) {
  const statusBar = _createStatusBar(container, client);
  container.appendChild(statusBar);

  const el = document.createElement("div");
  el.className = "sdp-graph";
  container.appendChild(el);

  let pollTimer = null;

  function render(state) {
    el.innerHTML = "";

    // Workflow state header
    _section(el, "Workflow State");
    const stats = document.createElement("div");
    stats.className = "sdp-graph__stats";

    if (!state.has_workflow) {
      stats.innerHTML = `<span style="color:var(--cz-text-muted)">No workflow loaded</span>`;
      el.appendChild(stats);
      return;
    }

    const layerCounts = _countOpinions(state.deltas);
    const layerSummary = Object.entries(layerCounts)
      .map(([op, n]) => `${n}${op}`)
      .join(", ");

    stats.innerHTML = `
      <span>Layers: <span class="sdp-graph__stat-value">${state.delta_count}${layerSummary ? ` (${layerSummary})` : ""}</span></span>
      <span>Integrity: ${
        state.integrity
          ? state.integrity.intact
            ? `<span class="sdp-integrity--ok">\u2713 verified</span>`
            : `<span class="sdp-integrity--fail">\u2717 tampered</span>`
          : `<span style="color:var(--cz-text-muted)">\u2014</span>`
      }</span>
    `;
    el.appendChild(stats);

    // Node cards
    if (state.nodes && Object.keys(state.nodes).length > 0) {
      const nodesSection = document.createElement("div");
      nodesSection.style.display = "flex";
      nodesSection.style.flexDirection = "column";
      nodesSection.style.gap = "var(--cz-2)";
      nodesSection.style.marginTop = "var(--cz-4)";

      for (const [nid, node] of Object.entries(state.nodes)) {
        nodesSection.appendChild(_nodeCard(nid, node, state.deltas, client));
      }
      el.appendChild(nodesSection);
    }

    // Delta history
    if (state.deltas && state.deltas.length > 0) {
      const deltaSection = document.createElement("div");
      deltaSection.style.marginTop = "var(--cz-5)";
      _section(deltaSection, "Delta History");

      for (const delta of [...state.deltas].reverse()) {
        deltaSection.appendChild(_deltaRow(delta));
      }

      // Controls
      const controls = document.createElement("div");
      controls.style.display = "flex";
      controls.style.gap = "var(--cz-2)";
      controls.style.marginTop = "var(--cz-3)";

      const rollbackBtn = document.createElement("button");
      rollbackBtn.className = "sdp-btn";
      rollbackBtn.textContent = "Undo last";
      rollbackBtn.addEventListener("click", async () => {
        await client.rollback();
        _refresh();
      });
      controls.appendChild(rollbackBtn);

      deltaSection.appendChild(controls);
      el.appendChild(deltaSection);
    }
  }

  async function _refresh() {
    try {
      const state = await client.getGraphState();
      render(state);
    } catch (e) {
      el.innerHTML = `<div style="color:var(--cz-text-muted);padding:var(--cz-4)">Could not load graph state</div>`;
    }
  }

  function startPolling() {
    _refresh();
    _refreshStatusBar(statusBar, client);
    pollTimer = setInterval(_refresh, 2000);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  return { startPolling, stopPolling, refresh: _refresh };
}

/* ── Helpers ───────────────────────────────────────────────────── */

function _section(parent, label) {
  const lbl = document.createElement("div");
  lbl.className = "sdp-section__label";
  lbl.textContent = label;
  parent.appendChild(lbl);

  const rule = document.createElement("div");
  rule.className = "sdp-section__rule";
  parent.appendChild(rule);
}

function _countOpinions(deltas) {
  const counts = {};
  for (const d of deltas || []) {
    counts[d.opinion] = (counts[d.opinion] || 0) + 1;
  }
  return counts;
}

function _nodeCard(nid, node, deltas, client) {
  const card = document.createElement("div");
  card.className = "sdp-node";

  const header = document.createElement("div");
  header.className = "sdp-node__header";
  header.textContent = `${nid}: ${node.class_type}`;
  card.appendChild(header);

  const inputs = node.inputs || {};
  const opinionMap = _buildOpinionMap(nid, deltas);

  for (const [key, val] of Object.entries(inputs)) {
    const row = document.createElement("div");
    row.className = "sdp-node__row";

    const keyEl = document.createElement("span");
    keyEl.className = "sdp-node__key";
    keyEl.textContent = key;
    row.appendChild(keyEl);

    const valEl = document.createElement("span");
    // Check if link
    if (Array.isArray(val) && val.length === 2 && typeof val[0] === "string" && typeof val[1] === "number") {
      valEl.className = "sdp-node__val sdp-node__val--link";
      valEl.textContent = `\u2190 [${val[0]}]`;
      row.appendChild(valEl);
      // No edit for connections
      row.appendChild(document.createElement("span"));
    } else {
      valEl.className = "sdp-node__val";
      const displayVal = typeof val === "string" && val.length > 20 ? val.slice(0, 20) + "..." : String(val);
      valEl.textContent = displayVal;

      // Opinion badge
      const op = opinionMap[key];
      if (op) {
        const badge = document.createElement("span");
        badge.className = "sdp-opinion";
        badge.textContent = `(${op})`;
        badge.style.color = `var(--cz-opinion-${op})`;
        valEl.appendChild(badge);
      }

      row.appendChild(valEl);

      // Edit button
      const editBtn = document.createElement("button");
      editBtn.className = "sdp-node__edit";
      editBtn.textContent = "edit";
      editBtn.addEventListener("click", () => _startEdit(row, nid, key, val, client));
      row.appendChild(editBtn);
    }

    card.appendChild(row);
  }

  return card;
}

function _buildOpinionMap(nid, deltas) {
  // Build map of param -> highest opinion that set it
  const map = {};
  const priorityOrder = { P: 1, R: 2, V: 3, I: 4, L: 5, S: 6 };
  for (const d of deltas || []) {
    if (d.mutations && d.mutations[nid]) {
      for (const param of Object.keys(d.mutations[nid])) {
        const existing = map[param];
        if (!existing || priorityOrder[d.opinion] > priorityOrder[existing]) {
          map[param] = d.opinion;
        }
      }
    }
  }
  return map;
}

function _startEdit(row, nid, key, currentVal, client) {
  const valCell = row.querySelector(".sdp-node__val");
  const editBtn = row.querySelector(".sdp-node__edit");
  if (!valCell || !editBtn) return;

  const input = document.createElement("input");
  input.className = "sdp-node__edit-input";
  input.value = String(currentVal);
  input.type = "text";

  const originalHTML = valCell.innerHTML;
  valCell.innerHTML = "";
  valCell.appendChild(input);
  editBtn.style.display = "none";
  input.focus();
  input.select();

  async function commit() {
    let value = input.value;
    // Try to parse as number
    if (!isNaN(value) && value.trim() !== "") {
      value = Number(value);
    }
    await client.setInput(nid, key, value);
    cleanup();
  }

  function cancel() {
    valCell.innerHTML = originalHTML;
    editBtn.style.display = "";
  }

  function cleanup() {
    // Refresh will rebuild the card
    input.removeEventListener("keydown", onKey);
    input.removeEventListener("blur", onBlur);
  }

  function onKey(e) {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    if (e.key === "Escape") { cancel(); }
  }

  function onBlur() { cancel(); }

  input.addEventListener("keydown", onKey);
  input.addEventListener("blur", onBlur);
}

function _deltaRow(delta) {
  const wrapper = document.createElement("div");

  const row = document.createElement("div");
  row.className = "sdp-delta";

  const chevron = document.createElement("span");
  chevron.textContent = "\u25B8";
  chevron.style.fontSize = "10px";
  chevron.style.transition = "transform var(--cz-ease)";
  row.appendChild(chevron);

  const opinion = document.createElement("span");
  opinion.className = "sdp-delta__opinion";
  opinion.textContent = delta.opinion;
  opinion.style.color = `var(--cz-opinion-${delta.opinion})`;
  row.appendChild(opinion);

  const desc = document.createElement("span");
  desc.className = "sdp-delta__desc";
  desc.textContent = delta.description || delta.layer_id;
  row.appendChild(desc);

  const time = document.createElement("span");
  time.className = "sdp-delta__time";
  time.textContent = _relativeTime(delta.timestamp);
  row.appendChild(time);

  const body = document.createElement("div");
  body.className = "sdp-delta__body";
  body.textContent = JSON.stringify(delta.mutations, null, 2);

  row.addEventListener("click", () => {
    const expanded = wrapper.classList.toggle("sdp-delta--expanded");
    chevron.style.transform = expanded ? "rotate(90deg)" : "";
  });

  wrapper.appendChild(row);
  wrapper.appendChild(body);
  return wrapper;
}

function _relativeTime(ts) {
  const diff = (Date.now() / 1000) - ts;
  if (diff < 60) return "now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

/* ── Status Bar ─────────────────────────────────────────────────── */

function _createStatusBar(container, client) {
  const bar = document.createElement("div");
  bar.className = "sdp-status";
  bar.innerHTML = `<span class="sdp-status__loading">Checking workflow health...</span>`;
  return bar;
}

async function _refreshStatusBar(bar, client) {
  bar.innerHTML = "";
  const warnings = [];

  try {
    const [wiringRes, deprecRes] = await Promise.allSettled([
      fetch("/superduper-panel/suggest-wiring").then((r) => r.json()),
      fetch("/superduper-panel/check-deprecations").then((r) => r.json()),
    ]);

    // Process wiring issues
    if (wiringRes.status === "fulfilled" && wiringRes.value) {
      const wiring = wiringRes.value;
      const missing = wiring.missing_nodes || wiring.missing || [];
      if (missing.length > 0) {
        warnings.push({
          text: `${missing.length} missing node${missing.length > 1 ? "s" : ""}`,
          action: "Repair",
          handler: async (btn) => {
            btn.disabled = true;
            btn.textContent = "...";
            try {
              await client.repairWorkflow(true);
              btn.textContent = "Done";
              _refreshStatusBar(bar, client);
            } catch { btn.textContent = "Failed"; }
          },
        });
      }
      const issues = wiring.issues || wiring.warnings || [];
      for (const issue of issues) {
        const text = typeof issue === "string" ? issue : issue.message || issue.description || JSON.stringify(issue);
        warnings.push({ text });
      }
    }

    // Process deprecation issues
    if (deprecRes.status === "fulfilled" && deprecRes.value) {
      const deprec = deprecRes.value;
      const deprecated = deprec.deprecated || deprec.nodes || [];
      if (deprecated.length > 0) {
        warnings.push({
          text: `${deprecated.length} deprecated node${deprecated.length > 1 ? "s" : ""}`,
          action: "Migrate",
          handler: async (btn) => {
            btn.disabled = true;
            btn.textContent = "...";
            try {
              await fetch("/superduper-panel/migrate-deprecated", { method: "POST" });
              btn.textContent = "Done";
              _refreshStatusBar(bar, client);
            } catch { btn.textContent = "Failed"; }
          },
        });
      }
    }
  } catch {
    // Network error — skip status bar
  }

  if (warnings.length === 0) {
    const ok = document.createElement("div");
    ok.className = "sdp-status__ok";
    ok.textContent = "\u2713 Workflow healthy";
    bar.appendChild(ok);
    return;
  }

  for (const warn of warnings) {
    const row = document.createElement("div");
    row.className = "sdp-status__warning";

    const text = document.createElement("span");
    text.textContent = warn.text;
    row.appendChild(text);

    if (warn.action && warn.handler) {
      const btn = document.createElement("button");
      btn.className = "sdp-status__action";
      btn.textContent = warn.action;
      btn.addEventListener("click", () => warn.handler(btn));
      row.appendChild(btn);
    }

    const dismissBtn = document.createElement("button");
    dismissBtn.className = "sdp-status__dismiss";
    dismissBtn.textContent = "\u2715";
    dismissBtn.addEventListener("click", () => row.remove());
    row.appendChild(dismissBtn);

    bar.appendChild(row);
  }
}
