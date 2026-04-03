/* ── Agent Client — HTTP/SSE communication ─────────────────────── */

const BASE = "/superduper-panel";

export class AgentClient {
  constructor() {
    this._base = BASE;
  }

  async health() {
    const r = await fetch(`${this._base}/health`);
    return r.json();
  }

  async getGraphState() {
    const r = await fetch(`${this._base}/graph-state`);
    return r.json();
  }

  async getWorkflowApi() {
    const r = await fetch(`${this._base}/get-workflow-api`);
    if (!r.ok) return null;
    const data = await r.json();
    return data.workflow || null;
  }

  async setInput(nodeId, inputName, value) {
    const r = await fetch(`${this._base}/set-input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: nodeId, input_name: inputName, value }),
    });
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfycozy:workflow-changed"));
      document.dispatchEvent(new CustomEvent("superduper:node_touch", {
        detail: { nodeId: String(nodeId) },
      }));
    }
    return result;
  }

  async addNode(classType, inputs) {
    const r = await fetch(`${this._base}/add-node`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_type: classType, inputs: inputs || {} }),
    });
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfycozy:workflow-changed"));
    }
    return result;
  }

  async connectNodes(srcId, srcSlot, dstId, dstSlot) {
    const r = await fetch(`${this._base}/connect-nodes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_id: srcId, source_slot: srcSlot,
        target_id: dstId, target_slot: dstSlot,
      }),
    });
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfycozy:workflow-changed"));
      document.dispatchEvent(new CustomEvent("superduper:node_touch", {
        detail: { nodeId: String(srcId) },
      }));
      document.dispatchEvent(new CustomEvent("superduper:node_touch", {
        detail: { nodeId: String(dstId) },
      }));
    }
    return result;
  }

  async applyPatch(patches) {
    const r = await fetch(`${this._base}/apply-patch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patches }),
    });
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfycozy:workflow-changed"));
    }
    return result;
  }

  async rollback() {
    const r = await fetch(`${this._base}/rollback`, { method: "POST" });
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfycozy:workflow-changed"));
    }
    return result;
  }

  async reset() {
    const r = await fetch(`${this._base}/reset`, { method: "POST" });
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfycozy:workflow-changed"));
    }
    return result;
  }

  async getExperience() {
    const r = await fetch(`${this._base}/experience`);
    return r.json();
  }

  async getAutoresearch() {
    const r = await fetch(`${this._base}/autoresearch`);
    return r.json();
  }

  async applyPrediction(predictionId, path) {
    const r = await fetch(`${this._base}/prediction/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prediction_id: predictionId, path }),
    });
    return r.json();
  }

  async ignorePrediction(predictionId) {
    const r = await fetch(`${this._base}/prediction/ignore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prediction_id: predictionId }),
    });
    return r.json();
  }

  // ── Discovery ────────────────────────────────────────────────

  async discover(query, opts = {}) {
    const r = await fetch(`${this._base}/discover`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, ...opts }),
    });
    return r.json();
  }

  // ── Provisioning ─────────────────────────────────────────────

  async installNodePack(url, name) {
    const body = { url };
    if (name) body.name = name;
    const r = await fetch(`${this._base}/install-node-pack`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.json();
  }

  async downloadModel(url, modelType, filename) {
    const body = { url };
    if (modelType) body.model_type = modelType;
    if (filename) body.filename = filename;
    const r = await fetch(`${this._base}/download-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.json();
  }

  // ── Repair ───────────────────────────────────────────────────

  async repairWorkflow(autoInstall = true) {
    const r = await fetch(`${this._base}/repair-workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto_install: autoInstall }),
    });
    return r.json();
  }

  // ── Workflow Persistence ─────────────────────────────────────

  async saveWorkflow(path) {
    const r = await fetch(`${this._base}/save-workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    return r.json();
  }

  // ── Session ──────────────────────────────────────────────────

  async listSessions() {
    const r = await fetch(`${this._base}/list-sessions`);
    return r.json();
  }

  async saveSession(name) {
    const r = await fetch(`${this._base}/save-session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    return r.json();
  }

  async loadSessionData(name) {
    const r = await fetch(`${this._base}/load-session-data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    return r.json();
  }

  // ── CivitAI ──────────────────────────────────────────────────

  async getTrendingModels(opts = {}) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(opts)) {
      if (v != null) params.set(k, v);
    }
    const qs = params.toString();
    const r = await fetch(`${this._base}/trending-models${qs ? "?" + qs : ""}`);
    return r.json();
  }

  // ── Model Compatibility ──────────────────────────────────────

  async checkCompatibility(models) {
    const r = await fetch(`${this._base}/check-compatibility`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models }),
    });
    return r.json();
  }
}
