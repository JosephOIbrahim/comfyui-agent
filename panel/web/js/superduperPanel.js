/* ── Comfy Cozy Panel — Headless Canvas Bridge ─────────────────────
 *  All visible UI lives in the native left sidebar (ui/ extension).
 *  This module is what keeps the canvas and the agent in sync:
 *
 *    Canvas → Agent : on graph change, POST API workflow to backend
 *    Agent  → Canvas: on workflow-changed event, push literal widget values
 *    Agent  → Canvas: on node_touch event, briefly highlight a node
 *
 *  No DOM is mounted by this file. No fonts loaded. No buttons rendered.
 * ────────────────────────────────────────────────────────────────── */

import { app } from "../../../scripts/app.js";
import { AgentClient } from "./agentClient.js";

const client = new AgentClient();

/* ── Canvas → Agent Sync ──────────────────────────────────────── */

let _lastGraphHash = null;

/**
 * Read the live ComfyUI canvas and inject into agent workflow state.
 * Without this bridge, the agent can't see what's on the user's canvas.
 */
async function syncCanvasToAgent() {
  try {
    if (!app.graph) return;

    const graphData = app.graph.serialize();
    if (!graphData) return;

    // Cheap change detection — skip if structure hasn't changed
    const hash = JSON.stringify(graphData).length;
    if (hash === _lastGraphHash) return;
    _lastGraphHash = hash;

    // Use ComfyUI's own graphToPrompt to get API format
    let apiWorkflow = null;
    try {
      const prompt = await app.graphToPrompt();
      if (prompt && prompt.output) {
        apiWorkflow = prompt.output;
      }
    } catch (e) {
      apiWorkflow = graphData; // fallback: let backend parse UI format
    }

    if (!apiWorkflow) return;

    await fetch("/comfy-cozy/load-workflow-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: apiWorkflow, source: "<canvas>" }),
    });
  } catch (e) {
    // Silent — sync is best-effort
    console.debug("[Comfy Cozy] Canvas sync failed:", e);
  }
}

async function setupCanvasSync() {
  // Initial sync after the graph settles
  setTimeout(syncCanvasToAgent, 1000);

  if (app.graph) {
    const origOnChange = app.graph.onAfterChange;
    app.graph.onAfterChange = function () {
      if (origOnChange) origOnChange.apply(this, arguments);
      clearTimeout(app.graph._sdpSyncTimer);
      app.graph._sdpSyncTimer = setTimeout(syncCanvasToAgent, 500);
    };
  }

  // Re-sync after execution completes (outputs may have changed)
  try {
    const { api } = await import("../../../scripts/api.js");
    if (api) {
      api.addEventListener("executed", () => {
        setTimeout(syncCanvasToAgent, 200);
      });
    }
  } catch (e) {
    // api import unavailable — non-fatal
  }
}

/* ── Agent → Canvas Push ──────────────────────────────────────── */

/**
 * Fetch the current agent workflow and push literal input values
 * onto the live ComfyUI canvas.  Only updates widget values that
 * differ; connections (arrays) are left untouched.
 */
async function pushAgentToCanvas() {
  try {
    const workflow = await client.getWorkflowApi();
    if (!workflow || !app.graph) return;

    for (const [nodeId, nodeData] of Object.entries(workflow)) {
      if (!nodeData || !nodeData.inputs) continue;
      const node = app.graph.getNodeById(parseInt(nodeId, 10));
      if (!node || !node.widgets) continue;

      for (const widget of node.widgets) {
        const apiValue = nodeData.inputs[widget.name];
        if (apiValue !== undefined && !Array.isArray(apiValue)) {
          if (widget.value !== apiValue) {
            widget.value = apiValue;
          }
        }
      }
    }

    app.canvas.setDirty(true, true);
  } catch (e) {
    console.debug("[Comfy Cozy] Canvas push failed:", e);
  }
}

/**
 * Briefly flash a node on the canvas after the agent touches it.
 */
function highlightNode(nodeId) {
  if (!app.graph) return;
  const node = app.graph.getNodeById(parseInt(nodeId, 10));
  if (!node) return;

  const prev = node.color;
  node.color = "#7c3aed";              // vivid purple flash
  app.canvas.setDirty(true, true);

  setTimeout(() => {
    node.color = prev;
    app.canvas.setDirty(true, true);
  }, 600);
}

document.addEventListener("comfy-cozy:workflow-changed", () => {
  pushAgentToCanvas();
});

document.addEventListener("comfy-cozy:node_touch", (e) => {
  const nid = e.detail?.nodeId;
  if (nid) highlightNode(nid);
});

/* ── Register Extension (headless) ────────────────────────────── */

app.registerExtension({
  name: "comfy-cozy.panel",
  async setup() {
    // Health check — informational only
    try {
      await client.health();
      console.log("[Comfy Cozy] Canvas bridge connected to agent backend");
    } catch (e) {
      console.log("[Comfy Cozy] Agent backend not available — canvas bridge idle");
    }

    setupCanvasSync();
  },
});
