/* ── APP Mode — Chat Interface ──────────────────────────────────── */

import { createPredictionCard } from "./predictionOverlay.js";

export function createAppMode(container, client) {
  const messagesEl = document.createElement("div");
  messagesEl.className = "sdp-messages";
  messagesEl.setAttribute("role", "log");
  messagesEl.setAttribute("aria-live", "polite");

  const inputBar = document.createElement("div");
  inputBar.className = "sdp-input-bar";

  const textarea = document.createElement("textarea");
  textarea.className = "sdp-input";
  textarea.placeholder = "Ask about your workflow...";
  textarea.rows = 1;

  const sendBtn = document.createElement("button");
  sendBtn.className = "sdp-send";
  sendBtn.textContent = "\u2192";
  sendBtn.setAttribute("aria-label", "Send message");

  inputBar.appendChild(textarea);
  inputBar.appendChild(sendBtn);

  const actionsBar = _createQuickActions(container, client, _appendSystemMessage);

  container.appendChild(messagesEl);
  container.appendChild(actionsBar);
  container.appendChild(inputBar);

  // State
  let busy = false;
  const history = _loadHistory();

  // Render saved history
  if (history.length === 0) {
    _addSystem("What would you like to do with your workflow?");
  } else {
    for (const msg of history) {
      _renderMessage(msg.role, msg.text, false);
    }
    _scrollToBottom();
  }

  // Input handling
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      _send();
    }
  });
  sendBtn.addEventListener("click", _send);

  // Auto-resize textarea
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 96) + "px";
  });

  function _send() {
    const text = textarea.value.trim();
    if (!text || busy) return;

    _addUser(text);
    textarea.value = "";
    textarea.style.height = "auto";
    _setBusy(true);

    // Check for view commands
    if (text.startsWith("/experience") || text.startsWith("/research")) {
      // These are handled by panel shell via custom event
      const event = new CustomEvent("sdp-command", { detail: { command: text.slice(1).split(" ")[0] } });
      container.dispatchEvent(event);
      _setBusy(false);
      return;
    }

    _showTyping();
    // Agent communication would happen here via SSE
    // For now, show a placeholder response
    setTimeout(() => {
      _clearTyping();
      _addAgent("Agent connected. Waiting for backend wiring.");
      _setBusy(false);
    }, 500);
  }

  function _setBusy(state) {
    busy = state;
    sendBtn.disabled = state;
    textarea.disabled = state;
    sendBtn.textContent = state ? "..." : "\u2192";
  }

  function _addUser(text) {
    _renderMessage("user", text);
    _saveMessage("user", text);
  }

  function _addAgent(text) {
    _renderMessage("agent", text);
    _saveMessage("agent", text);
  }

  function _addSystem(text) {
    _renderMessage("system", text);
  }

  let typingEl = null;

  function _showTyping() {
    if (typingEl) return;
    typingEl = document.createElement("div");
    typingEl.className = "sdp-msg";
    typingEl.innerHTML = `
      <span class="sdp-msg__label">Agent</span>
      <div class="sdp-typing">
        <span class="sdp-typing__dot"></span>
        <span class="sdp-typing__dot"></span>
        <span class="sdp-typing__dot"></span>
      </div>
    `;
    messagesEl.appendChild(typingEl);
    _scrollToBottom();
  }

  function _clearTyping() {
    if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  function _renderMessage(role, text, scroll = true) {
    const msg = document.createElement("div");
    msg.className = `sdp-msg sdp-msg--${role}`;

    if (role !== "system") {
      const label = document.createElement("span");
      label.className = "sdp-msg__label";
      label.textContent = role === "user" ? "You" : "Agent";
      msg.appendChild(label);
    }

    const body = document.createElement("div");
    body.className = "sdp-msg__body";

    if (role === "agent") {
      body.innerHTML = _renderMarkdown(text);  // sanitized via _sanitizeHtml
    } else {
      body.textContent = text;
    }

    msg.appendChild(body);
    messagesEl.appendChild(msg);

    if (scroll) _scrollToBottom();
    return msg;
  }

  function _scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // Simple markdown rendering (bold, italic, code, code blocks)
  // Safety chain: _esc() converts all HTML to entities first, then regex
  // produces only allowlisted tags. _sanitizeHtml() strips anything else
  // as a defense-in-depth measure before innerHTML assignment.
  function _renderMarkdown(text) {
    let html = _esc(text);
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // Line breaks
    html = html.replace(/\n/g, "<br>");
    return _sanitizeHtml(html);
  }

  function _sanitizeHtml(html) {
    // Allow only safe formatting tags produced by _renderMarkdown.
    // Strips any tag not in the allowlist (defense-in-depth against XSS).
    const ALLOWED = /^(pre|code|strong|em|br|p|span)$/i;
    return html.replace(/<\/?([a-zA-Z][a-zA-Z0-9]*)[^>]*>/g, (match, tag) => {
      return ALLOWED.test(tag) ? match : "";
    });
  }

  function _esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // LocalStorage persistence
  function _loadHistory() {
    try {
      const raw = localStorage.getItem("sd-panel-history");
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  }

  function _saveMessage(role, text) {
    history.push({ role, text });
    if (history.length > 100) history.splice(0, history.length - 100);
    try {
      localStorage.setItem("sd-panel-history", JSON.stringify(history));
    } catch { /* quota exceeded */ }
  }

  // Public API for adding tool cards and predictions
  return {
    addToolCard(name, params) {
      const card = document.createElement("div");
      card.className = "sdp-tool";
      const header = document.createElement("div");
      header.className = "sdp-tool__header";
      header.innerHTML = `<span>${_esc(name)}</span><span class="sdp-tool__chevron">\u25B8</span>`;

      const body = document.createElement("div");
      body.className = "sdp-tool__body";
      for (const [k, v] of Object.entries(params)) {
        const row = document.createElement("div");
        row.className = "sdp-tool__row";
        row.innerHTML = `<span class="sdp-tool__key">${_esc(k)}</span><span>${_esc(String(v))}</span>`;
        body.appendChild(row);
      }

      header.addEventListener("click", () => card.classList.toggle("sdp-tool--expanded"));
      card.appendChild(header);
      card.appendChild(body);
      messagesEl.appendChild(card);
      _scrollToBottom();
    },

    addPrediction(prediction) {
      const card = createPredictionCard(prediction, {
        onApply: (p) => client.applyPrediction(p.id, p.paths[0]),
        onIgnore: (p) => client.ignorePrediction(p.id),
      });
      messagesEl.appendChild(card);
      _scrollToBottom();
    },

    showProgress(pct, nodeLabel) {
      let prog = messagesEl.querySelector(".sdp-progress");
      if (!prog) {
        prog = document.createElement("div");
        prog.className = "sdp-progress";
        prog.innerHTML = `
          <div class="sdp-progress__bar"><div class="sdp-progress__fill"></div></div>
          <div class="sdp-progress__label">
            <span class="sdp-progress__node"></span>
            <span class="sdp-progress__pct"></span>
          </div>
        `;
        messagesEl.appendChild(prog);
      }
      prog.querySelector(".sdp-progress__fill").style.width = `${pct}%`;
      prog.querySelector(".sdp-progress__node").textContent = nodeLabel || "";
      prog.querySelector(".sdp-progress__pct").textContent = `${Math.round(pct)}%`;
      _scrollToBottom();
    },

    hideProgress() {
      const prog = messagesEl.querySelector(".sdp-progress");
      if (prog) prog.remove();
    },

    addMessage: _addAgent,
    addSystemMessage: _addSystem,
  };

  function _appendSystemMessage(container_, text) {
    _addSystem(text);
  }
}

/* ── Quick Actions Bar ──────────────────────────────────────────── */

function _createQuickActions(container, client, appendMsg) {
  const bar = document.createElement("div");
  bar.className = "sdp-actions";

  const actions = [
    {
      label: "Repair",
      icon: "\u2695",
      title: "Auto-install missing nodes",
      action: () => client.repairWorkflow(true),
    },
    {
      label: "Save",
      icon: "\uD83D\uDCBE",
      title: "Save workflow to disk",
      action: () => _promptSaveWorkflow(client, appendMsg, container),
    },
    {
      label: "Browse",
      icon: "\uD83D\uDD0D",
      title: "Browse & download models",
      action: () => { _openModelBrowser(container, client, appendMsg); return null; },
    },
    {
      label: "Wiring",
      icon: "\u26A1",
      title: "Show model wiring",
      action: () => _fetchWiring(client),
    },
  ];

  for (const act of actions) {
    const btn = document.createElement("button");
    btn.className = "sdp-action-btn";
    btn.title = act.title;
    btn.textContent = act.label;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const result = await act.action();
        if (result != null) {
          const text = typeof result === "string" ? result : JSON.stringify(result, null, 2);
          appendMsg(container, text);
        }
      } catch (e) {
        appendMsg(container, `Error: ${e.message}`);
      } finally {
        btn.disabled = false;
      }
    });
    bar.appendChild(btn);
  }

  return bar;
}

async function _promptSaveWorkflow(client, appendMsg, container) {
  const path = prompt("Save workflow to path:");
  if (!path) return null;
  const result = await client.saveWorkflow(path);
  return result;
}

async function _fetchWiring(client) {
  const r = await fetch("/superduper-panel/suggest-wiring");
  return r.json();
}

/* ── Model Browser ──────────────────────────────────────────────── */

function _openModelBrowser(container, client, appendMsg) {
  // Remove existing browser if open
  const existing = container.querySelector(".sdp-browser");
  if (existing) { existing.remove(); return; }

  const overlay = document.createElement("div");
  overlay.className = "sdp-browser";

  // Header
  const header = document.createElement("div");
  header.className = "sdp-browser__header";

  const title = document.createElement("span");
  title.className = "sdp-browser__title";
  title.textContent = "Model Browser";
  header.appendChild(title);

  const closeBtn = document.createElement("button");
  closeBtn.className = "sdp-header__btn";
  closeBtn.textContent = "\u2715";
  closeBtn.addEventListener("click", () => overlay.remove());
  header.appendChild(closeBtn);

  overlay.appendChild(header);

  // Search bar
  const searchRow = document.createElement("div");
  searchRow.className = "sdp-browser__search";

  const searchInput = document.createElement("input");
  searchInput.className = "sdp-browser__input";
  searchInput.type = "text";
  searchInput.placeholder = "Search models...";
  searchRow.appendChild(searchInput);

  const sourceSelect = document.createElement("select");
  sourceSelect.className = "sdp-browser__select";
  for (const src of ["All", "civitai", "huggingface", "registry"]) {
    const opt = document.createElement("option");
    opt.value = src === "All" ? "" : src;
    opt.textContent = src;
    sourceSelect.appendChild(opt);
  }
  searchRow.appendChild(sourceSelect);

  const searchBtn = document.createElement("button");
  searchBtn.className = "sdp-action-btn";
  searchBtn.textContent = "Search";
  searchRow.appendChild(searchBtn);

  overlay.appendChild(searchRow);

  // Results
  const results = document.createElement("div");
  results.className = "sdp-browser__results";
  results.innerHTML = `<div class="sdp-browser__empty">Search for models above</div>`;
  overlay.appendChild(results);

  // Search handler
  async function doSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    results.innerHTML = `<div class="sdp-browser__empty">Searching...</div>`;
    searchBtn.disabled = true;
    try {
      const opts = {};
      const source = sourceSelect.value;
      if (source) opts.source = source;
      const data = await client.discover(query, opts);
      _renderBrowserResults(results, data, client, appendMsg, container);
    } catch (e) {
      results.innerHTML = `<div class="sdp-browser__empty">Error: ${_escText(e.message)}</div>`;
    } finally {
      searchBtn.disabled = false;
    }
  }

  searchBtn.addEventListener("click", doSearch);
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); doSearch(); }
  });

  // Insert before the input bar
  const inputBar = container.querySelector(".sdp-input-bar");
  if (inputBar) {
    container.insertBefore(overlay, inputBar);
  } else {
    container.appendChild(overlay);
  }

  searchInput.focus();
}

function _renderBrowserResults(resultsEl, data, client, appendMsg, container) {
  resultsEl.innerHTML = "";
  const items = data.results || data.items || [];
  if (items.length === 0) {
    resultsEl.innerHTML = `<div class="sdp-browser__empty">No results found</div>`;
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "sdp-browser__item";

    const info = document.createElement("div");
    info.className = "sdp-browser__item-info";

    const name = document.createElement("span");
    name.className = "sdp-browser__item-name";
    name.textContent = item.name || item.title || "Unknown";
    info.appendChild(name);

    const meta = document.createElement("span");
    meta.className = "sdp-browser__item-meta";
    const parts = [];
    if (item.source) parts.push(item.source);
    if (item.type) parts.push(item.type);
    if (item.installed) parts.push("installed");
    meta.textContent = parts.join(" \u00B7 ");
    info.appendChild(meta);

    row.appendChild(info);

    if (item.installed) {
      const badge = document.createElement("span");
      badge.className = "sdp-browser__badge";
      badge.textContent = "Installed";
      row.appendChild(badge);
    } else if (item.url || item.download_url) {
      const dlBtn = document.createElement("button");
      dlBtn.className = "sdp-action-btn";
      dlBtn.textContent = "Install";
      dlBtn.addEventListener("click", async () => {
        dlBtn.disabled = true;
        dlBtn.textContent = "...";
        try {
          const url = item.download_url || item.url;
          let result;
          if (item.type === "node_pack" || item.type === "custom_node") {
            result = await client.installNodePack(url, item.name);
          } else {
            result = await client.downloadModel(url, item.model_type, item.filename);
          }
          dlBtn.textContent = "Done";
          appendMsg(container, `Installed: ${item.name || url}`);
        } catch (e) {
          dlBtn.textContent = "Fail";
          appendMsg(container, `Install failed: ${e.message}`);
        }
      });
      row.appendChild(dlBtn);
    }

    resultsEl.appendChild(row);
  }
}

function _escText(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
