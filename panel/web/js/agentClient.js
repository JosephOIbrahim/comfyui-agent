/* ── Agent Client — HTTP/SSE communication ─────────────────────── */

const BASE = "/comfy-cozy";

export class AgentClient {
  constructor() {
    this._base = BASE;
  }

  async health() {
    const r = await fetch(`${this._base}/health`);
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  async getGraphState() {
    const r = await fetch(`${this._base}/graph-state`);
    if (!r.ok) return { error: `Request failed: ${r.status}` };
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
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfy-cozy:workflow-changed"));
      document.dispatchEvent(new CustomEvent("comfy-cozy:node_touch", {
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
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfy-cozy:workflow-changed"));
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
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfy-cozy:workflow-changed"));
      document.dispatchEvent(new CustomEvent("comfy-cozy:node_touch", {
        detail: { nodeId: String(srcId) },
      }));
      document.dispatchEvent(new CustomEvent("comfy-cozy:node_touch", {
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
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfy-cozy:workflow-changed"));
    }
    return result;
  }

  async rollback() {
    const r = await fetch(`${this._base}/rollback`, { method: "POST" });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfy-cozy:workflow-changed"));
    }
    return result;
  }

  async reset() {
    const r = await fetch(`${this._base}/reset`, { method: "POST" });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    const result = await r.json();
    if (!result.error) {
      document.dispatchEvent(new Event("comfy-cozy:workflow-changed"));
    }
    return result;
  }

  async getExperience() {
    const r = await fetch(`${this._base}/experience`);
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  async getAutoresearch() {
    const r = await fetch(`${this._base}/autoresearch`);
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  async applyPrediction(predictionId, path) {
    // Endpoint not yet implemented server-side
    return { error: "Not yet implemented" };
  }

  async ignorePrediction(predictionId) {
    // Endpoint not yet implemented server-side
    return { error: "Not yet implemented" };
  }

  // ── Discovery ────────────────────────────────────────────────

  async discover(query, opts = {}) {
    const r = await fetch(`${this._base}/discover`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, ...opts }),
    });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
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
    if (!r.ok) return { error: `Request failed: ${r.status}` };
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
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  // ── Repair ───────────────────────────────────────────────────

  async repairWorkflow(autoInstall = true) {
    const r = await fetch(`${this._base}/repair-workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto_install: autoInstall }),
    });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  // ── Workflow Persistence ─────────────────────────────────────

  async saveWorkflow(path) {
    const r = await fetch(`${this._base}/save-workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  // ── Session ──────────────────────────────────────────────────

  async listSessions() {
    const r = await fetch(`${this._base}/list-sessions`);
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  async saveSession(name) {
    const r = await fetch(`${this._base}/save-session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  async loadSessionData(name) {
    const r = await fetch(`${this._base}/load-session-data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  // ── WebSocket (chat streaming) ────────────────────────────────

  connectWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/comfy-cozy/ws`;

    this._ws = new WebSocket(url);
    if (!this._wsListeners) this._wsListeners = {};  // preserve across reconnects
    this._wsReconnectTimer = null;

    this._ws.onopen = () => {
      console.debug("[Comfy Cozy] WebSocket connected");
    };

    this._ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        const listeners = this._wsListeners[data.type] || [];
        for (const fn of listeners) fn(data);
      } catch (e) {
        console.debug("[Comfy Cozy] Bad WS message:", evt.data);
      }
    };

    this._ws.onclose = () => {
      console.debug("[Comfy Cozy] WebSocket closed, reconnecting in 2s");
      this._wsReconnectTimer = setTimeout(() => this.connectWs(), 2000);
    };

    this._ws.onerror = () => {
      // onclose will fire after this, triggering reconnect
    };

    return this;
  }

  on(eventType, callback) {
    if (!this._wsListeners) this._wsListeners = {};
    if (!this._wsListeners[eventType]) this._wsListeners[eventType] = [];
    this._wsListeners[eventType].push(callback);
    return this;
  }

  sendChat(text) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: "chat", content: text }));
    }
  }

  sendWorkflow(data) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: "workflow", data }));
    }
  }

  // ── CivitAI ──────────────────────────────────────────────────

  async getTrendingModels(opts = {}) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(opts)) {
      if (v != null) params.set(k, v);
    }
    const qs = params.toString();
    const r = await fetch(`${this._base}/trending-models${qs ? "?" + qs : ""}`);
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }

  // ── Model Compatibility ──────────────────────────────────────

  async checkCompatibility(models) {
    const r = await fetch(`${this._base}/check-compatibility`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models }),
    });
    if (!r.ok) return { error: `Request failed: ${r.status}` };
    return r.json();
  }
}
