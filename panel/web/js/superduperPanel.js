/* ── Comfy Cozy Panel — Entry Point ──────────────────────────────
 *  Registers with ComfyUI's extension system.
 *  Creates the pill button and panel shell.
 *  Manages mode switching between APP, GRAPH, Experience, Research.
 * ────────────────────────────────────────────────────────────────── */

import { app } from "../../../scripts/app.js";
import { AgentClient } from "./agentClient.js";
import { createAppMode } from "./appMode.js";
import { createGraphMode } from "./graphMode.js";
import { createExperienceDash } from "./experienceDash.js";
import { createAutoresearchMonitor } from "./autoresearchMonitor.js";

/* ── Load fonts & stylesheet ──────────────────────────────────── */

const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Nunito:wght@700;800&display=swap";
document.head.appendChild(fontLink);

const cssLink = document.createElement("link");
cssLink.rel = "stylesheet";
cssLink.href = new URL("./styles.css", import.meta.url).href;
document.head.appendChild(cssLink);

/* ── State ────────────────────────────────────────────────────── */

const client = new AgentClient();
let panel = null;
let pill = null;
let currentMode = localStorage.getItem("sd-panel-mode") || "app";
let isOpen = sessionStorage.getItem("sd-panel-open") === "true";

// Mode controllers
let appCtrl = null;
let graphCtrl = null;
let expCtrl = null;
let researchCtrl = null;

/* ── Pill Button ──────────────────────────────────────────────── */

function createPillButton() {
  pill = document.createElement("button");
  pill.className = "sdp-pill";
  pill.textContent = "SD";
  pill.setAttribute("aria-label", "Toggle Comfy Cozy Panel");
  pill.addEventListener("click", togglePanel);
  document.body.appendChild(pill);
}

/* ── Panel Shell ──────────────────────────────────────────────── */

function createPanelShell() {
  panel = document.createElement("div");
  panel.className = `sdp-panel${isOpen ? "" : " sdp-panel--hidden"}`;

  // Resize handle
  const resize = document.createElement("div");
  resize.className = "sdp-panel__resize";
  _setupResize(resize);
  panel.appendChild(resize);

  // Header
  const header = document.createElement("div");
  header.className = "sdp-header";

  // Brand title
  const brand = document.createElement("span");
  brand.className = "sdp-brand";
  brand.textContent = "Comfy Cozy";
  header.appendChild(brand);

  const tabs = [
    { id: "app", label: "APP" },
    { id: "graph", label: "GRAPH" },
  ];

  const tabEls = {};
  for (const tab of tabs) {
    const btn = document.createElement("button");
    btn.className = `sdp-tab${tab.id === currentMode ? " sdp-tab--active" : ""}`;
    btn.textContent = tab.label;
    btn.addEventListener("click", () => switchMode(tab.id));
    header.appendChild(btn);
    tabEls[tab.id] = btn;
  }

  // Queue Prompt button
  const queueBtn = document.createElement("button");
  queueBtn.className = "sdp-queue-btn";
  queueBtn.textContent = "Queue";
  queueBtn.setAttribute("aria-label", "Queue prompt in ComfyUI");
  queueBtn.addEventListener("click", () => _queuePrompt(queueBtn));
  header.appendChild(queueBtn);

  const spacer = document.createElement("div");
  spacer.className = "sdp-header__spacer";
  header.appendChild(spacer);

  const minimizeBtn = document.createElement("button");
  minimizeBtn.className = "sdp-header__btn";
  minimizeBtn.textContent = "\u2500";
  minimizeBtn.setAttribute("aria-label", "Minimize panel");
  minimizeBtn.addEventListener("click", togglePanel);
  header.appendChild(minimizeBtn);

  const closeBtn = document.createElement("button");
  closeBtn.className = "sdp-header__btn";
  closeBtn.textContent = "\u00d7";
  closeBtn.setAttribute("aria-label", "Close panel");
  closeBtn.addEventListener("click", togglePanel);
  header.appendChild(closeBtn);

  panel.appendChild(header);

  // Content area
  const content = document.createElement("div");
  content.className = "sdp-content";
  content.id = "sdp-content";
  panel.appendChild(content);

  document.body.appendChild(panel);

  // Restore width
  const savedWidth = localStorage.getItem("sd-panel-width");
  if (savedWidth) panel.style.width = savedWidth;

  // Initialize modes
  _initModes(content, tabEls);

  // Listen for view commands from APP mode
  content.addEventListener("sdp-command", (e) => {
    const cmd = e.detail?.command;
    if (cmd === "experience") switchMode("experience");
    else if (cmd === "research") switchMode("research");
    else if (cmd === "app") switchMode("app");
  });

  // Show initial mode
  switchMode(currentMode);

  function switchMode(mode) {
    currentMode = mode;
    localStorage.setItem("sd-panel-mode", mode);

    // Update tabs
    for (const [id, el] of Object.entries(tabEls)) {
      el.classList.toggle("sdp-tab--active", id === mode || (mode === "experience" && id === "app") || (mode === "research" && id === "app"));
    }

    // Hide all mode containers
    for (const child of content.children) {
      child.style.display = "none";
    }

    // Show active mode
    if (mode === "app" && appCtrl) {
      content.querySelector(".sdp-app-container").style.display = "flex";
    } else if (mode === "graph" && graphCtrl) {
      content.querySelector(".sdp-graph-container").style.display = "block";
      graphCtrl.startPolling();
    } else if (mode === "experience" && expCtrl) {
      content.querySelector(".sdp-experience-container").style.display = "block";
      expCtrl.refresh();
    } else if (mode === "research" && researchCtrl) {
      content.querySelector(".sdp-research-container").style.display = "block";
      researchCtrl.refresh();
    }

    // Stop graph polling when not in graph mode
    if (mode !== "graph" && graphCtrl) {
      graphCtrl.stopPolling();
    }
  }
}

function _initModes(content, tabEls) {
  // APP mode container
  const appContainer = document.createElement("div");
  appContainer.className = "sdp-app-container";
  appContainer.style.display = "flex";
  appContainer.style.flexDirection = "column";
  appContainer.style.height = "100%";
  content.appendChild(appContainer);
  appCtrl = createAppMode(appContainer, client);

  // GRAPH mode container
  const graphContainer = document.createElement("div");
  graphContainer.className = "sdp-graph-container";
  graphContainer.style.display = "none";
  content.appendChild(graphContainer);
  graphCtrl = createGraphMode(graphContainer, client);

  // Experience dashboard container
  const expContainer = document.createElement("div");
  expContainer.className = "sdp-experience-container";
  expContainer.style.display = "none";
  content.appendChild(expContainer);
  expCtrl = createExperienceDash(expContainer, client);

  // Autoresearch monitor container
  const researchContainer = document.createElement("div");
  researchContainer.className = "sdp-research-container";
  researchContainer.style.display = "none";
  content.appendChild(researchContainer);
  researchCtrl = createAutoresearchMonitor(researchContainer, client);
}

function togglePanel() {
  isOpen = !isOpen;
  sessionStorage.setItem("sd-panel-open", String(isOpen));
  if (panel) {
    panel.classList.toggle("sdp-panel--hidden", !isOpen);
  }
}

function _setupResize(handle) {
  let startX, startWidth;

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    startWidth = panel.offsetWidth;
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });

  function onMove(e) {
    const delta = startX - e.clientX;
    const newWidth = Math.min(600, Math.max(300, startWidth + delta));
    panel.style.width = newWidth + "px";
  }

  function onUp() {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    localStorage.setItem("sd-panel-width", panel.style.width);
  }
}

/* ── Queue Prompt ─────────────────────────────────────────────── */

async function _queuePrompt(btn) {
  const originalText = btn.textContent;
  try {
    btn.textContent = "Queuing\u2026";
    btn.classList.add("sdp-queue-btn--running");

    // Use ComfyUI's native queuePrompt — identical to the built-in button.
    // arg 0 = queue at end (positive = front, 0 = end)
    // arg 1 = batch count (default 1)
    await app.queuePrompt(0, 1);

    btn.textContent = "Queued";
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove("sdp-queue-btn--running");
    }, 1500);
  } catch (e) {
    console.error("[Comfy Cozy Panel] Queue prompt failed:", e);
    btn.textContent = "Error";
    btn.classList.remove("sdp-queue-btn--running");
    setTimeout(() => { btn.textContent = originalText; }, 2000);
  }
}

/* ── Canvas ↔ Agent Sync ──────────────────────────────────────── */

let _lastGraphHash = null;

/**
 * Read the live ComfyUI canvas and inject into agent workflow state.
 * This is the critical bridge: without it, the agent can't see or
 * modify the workflow on the canvas.
 */
async function syncCanvasToAgent() {
  try {
    if (!app.graph) return;

    // Serialize the current graph from LiteGraph
    const graphData = app.graph.serialize();
    if (!graphData) return;

    // Quick change detection — skip if graph hasn't changed
    const hash = JSON.stringify(graphData).length; // cheap proxy
    if (hash === _lastGraphHash) return;
    _lastGraphHash = hash;

    // Extract API format from the serialized graph
    // ComfyUI's graph.serialize() returns UI format with nodes array.
    // We need to extract the prompt (API format) from it.
    // The cleanest way: use ComfyUI's own graphToPrompt
    let apiWorkflow = null;
    try {
      const prompt = await app.graphToPrompt();
      if (prompt && prompt.output) {
        apiWorkflow = prompt.output;
      }
    } catch (e) {
      // Fallback: send the raw graph and let the backend parse it
      apiWorkflow = graphData;
    }

    if (!apiWorkflow) return;

    // POST to agent backend
    await fetch("/superduper-panel/load-workflow-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: apiWorkflow, source: "<canvas>" }),
    });

    console.log("[Comfy Cozy Panel] Canvas synced to agent state");
  } catch (e) {
    // Silent — sync is best-effort
    console.debug("[Comfy Cozy Panel] Canvas sync failed:", e);
  }
}

/**
 * Set up listeners for canvas changes.
 * ComfyUI fires events when the graph is modified.
 */
async function setupCanvasSync() {
  // Sync on initial load
  setTimeout(syncCanvasToAgent, 1000);

  // Sync when graph changes (node added/removed/connected)
  // ComfyUI uses LiteGraph which fires "change" on the graph
  if (app.graph) {
    const origOnChange = app.graph.onAfterChange;
    app.graph.onAfterChange = function () {
      if (origOnChange) origOnChange.apply(this, arguments);
      // Debounce: don't sync on every tiny change
      clearTimeout(app.graph._sdpSyncTimer);
      app.graph._sdpSyncTimer = setTimeout(syncCanvasToAgent, 500);
    };
  }

  // Also sync when execution completes (outputs may have changed)
  try {
    const { api } = await import("../../../scripts/api.js");
    if (api) {
      api.addEventListener("executed", () => {
        setTimeout(syncCanvasToAgent, 200);
      });
    }
  } catch (e) {
    // api import may not be available yet
  }
}

/* ── Agent → Canvas Reverse Bridge ────────────────────────────── */

/**
 * Fetch the current agent workflow and push literal input values
 * onto the live ComfyUI canvas.  Only updates widget values that
 * differ; connections (arrays) are left untouched.
 */
async function pushAgentToCanvas() {
  try {
    const workflow = await client.getWorkflowApi();
    if (!workflow) return;
    if (!app.graph) return;

    for (const [nodeId, nodeData] of Object.entries(workflow)) {
      if (!nodeData || !nodeData.inputs) continue;
      const node = app.graph.getNodeById(parseInt(nodeId, 10));
      if (!node || !node.widgets) continue;

      for (const widget of node.widgets) {
        const apiValue = nodeData.inputs[widget.name];
        // Only update literal values, skip connections (arrays)
        if (apiValue !== undefined && !Array.isArray(apiValue)) {
          if (widget.value !== apiValue) {
            widget.value = apiValue;
          }
        }
      }
    }

    app.canvas.setDirty(true, true);
    console.log("[Comfy Cozy Panel] Canvas updated from agent state");
  } catch (e) {
    console.debug("[Comfy Cozy Panel] Canvas push failed:", e);
  }
}

/**
 * Briefly highlight a node on the canvas after the agent touches it.
 * Draws a coloured outline for ~600ms then fades.
 */
function highlightNode(nodeId) {
  if (!app.graph) return;
  const node = app.graph.getNodeById(parseInt(nodeId, 10));
  if (!node) return;

  const prev = node.color;
  node.color = "#7c3aed";               // vivid purple flash
  app.canvas.setDirty(true, true);

  setTimeout(() => {
    node.color = prev;
    app.canvas.setDirty(true, true);
  }, 600);
}

// Listen for the event that mutation methods in agentClient fire
document.addEventListener("comfycozy:workflow-changed", () => {
  pushAgentToCanvas();
});

// Listen for per-node touch events
document.addEventListener("superduper:node_touch", (e) => {
  const nid = e.detail?.nodeId;
  if (nid) highlightNode(nid);
});

/* ── Register Extension ───────────────────────────────────────── */

app.registerExtension({
  name: "superduper.panel",
  async setup() {
    createPillButton();
    createPanelShell();

    // Health check
    try {
      await client.health();
      console.log("[Comfy Cozy Panel] Connected to agent backend");
    } catch (e) {
      console.log("[Comfy Cozy Panel] Agent backend not available — UI-only mode");
    }

    // Start canvas ↔ agent synchronization
    setupCanvasSync();

    // Listen for ComfyUI execution events
    try {
      const { api } = await import("../../../scripts/api.js");
      if (api) {
        api.addEventListener("execution_start", () => {
          if (appCtrl) appCtrl.showProgress(0, "Starting...");
        });
        api.addEventListener("progress", ({ detail }) => {
          if (appCtrl && detail) {
            const pct = detail.max > 0 ? (detail.value / detail.max) * 100 : 0;
            appCtrl.showProgress(pct, `Step ${detail.value}/${detail.max}`);
          }
        });
        api.addEventListener("executing", ({ detail }) => {
          if (detail && detail.node === null && appCtrl) {
            appCtrl.hideProgress();
          }
        });
      }
    } catch (e) {
      console.debug("[Comfy Cozy Panel] API events not available");
    }
  },
});
