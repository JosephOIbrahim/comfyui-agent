import { app } from "../../../scripts/app.js";

/* ── SUPER DUPER Sidebar ─────────────────────────────────────────────
 *  Registers a sidebar tab in ComfyUI's extension manager.
 *  Chat interface with WebSocket connection to agent brain.
 * ──────────────────────────────────────────────────────────────────── */

const SIDEBAR_ID = "superduper";
const SIDEBAR_TITLE = "SUPER DUPER";

/* ── Load stylesheet ──────────────────────────────────────────────── */

const cssLink = document.createElement("link");
cssLink.rel = "stylesheet";
cssLink.href = new URL("../css/superduper.css", import.meta.url).href;
document.head.appendChild(cssLink);

/* ── Message rendering ────────────────────────────────────────────── */

function createMessageEl(role, text) {
  const msg = document.createElement("div");
  msg.className = `sd-message sd-message--${role}`;

  if (role !== "system") {
    const label = document.createElement("span");
    label.className = "sd-message__label";
    label.textContent = role === "user" ? "You" : "Agent";
    msg.appendChild(label);
  }

  const body = document.createElement("span");
  body.className = "sd-message__body";
  body.textContent = text;
  msg.appendChild(body);

  return msg;
}

/* ── WebSocket connection ─────────────────────────────────────────── */

class AgentConnection {
  constructor(onMessage, onStatus) {
    this.ws = null;
    this.onMessage = onMessage;   // (event) => void
    this.onStatus = onStatus;     // (status) => void
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 15000;
    this.currentDelay = this.reconnectDelay;
    this._closed = false;
    this._streamEl = null;        // element accumulating streamed text
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/superduper/ws`;

    this.onStatus("connecting");
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.currentDelay = this.reconnectDelay;
      this.onStatus("connected");
    };

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        this.onMessage(data);
      } catch (e) {
        console.error("[SUPER DUPER] Bad message:", evt.data);
      }
    };

    this.ws.onclose = () => {
      this.onStatus("disconnected");
      if (!this._closed) {
        setTimeout(() => this.connect(), this.currentDelay);
        this.currentDelay = Math.min(this.currentDelay * 1.5, this.maxReconnectDelay);
      }
    };

    this.ws.onerror = (err) => {
      console.error("[SUPER DUPER] WebSocket error:", err);
    };
  }

  send(type, payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, ...payload }));
      return true;
    }
    return false;
  }

  close() {
    this._closed = true;
    if (this.ws) this.ws.close();
  }
}

/* ── Build the sidebar DOM ────────────────────────────────────────── */

function buildSidebar(el) {
  el.style.height = "100%";
  el.style.display = "flex";
  el.style.flexDirection = "column";
  el.style.fontFamily = "system-ui, -apple-system, sans-serif";

  el.innerHTML = `
    <div class="sd-header">
      <span class="sd-header__title">${SIDEBAR_TITLE}</span>
      <span class="sd-header__status" id="sd-status"></span>
    </div>
    <div class="sd-stage-bar" id="sd-stage" style="display:none">
      <span class="sd-stage__label" id="sd-stage-label"></span>
      <span class="sd-stage__detail" id="sd-stage-detail"></span>
    </div>
    <div class="sd-messages" id="sd-messages">
      <div class="sd-message sd-message--system">
        <span class="sd-message__body">Ready. Type below.</span>
      </div>
    </div>
    <div class="sd-input-bar">
      <input
        type="text"
        id="sd-input"
        class="sd-input"
        placeholder="Ask me anything about your workflow..."
        autocomplete="off"
      />
      <button id="sd-send" class="sd-send-btn">Send</button>
    </div>
  `;

  const messagesEl = el.querySelector("#sd-messages");
  const inputEl = el.querySelector("#sd-input");
  const sendBtn = el.querySelector("#sd-send");
  const statusEl = el.querySelector("#sd-status");
  const stageBar = el.querySelector("#sd-stage");
  const stageLabel = el.querySelector("#sd-stage-label");
  const stageDetail = el.querySelector("#sd-stage-detail");

  let busy = false;
  let streamingEl = null; // element receiving streamed text deltas

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setStage(stage, detail) {
    if (stage === "DONE" || !stage) {
      stageBar.style.display = "none";
      return;
    }
    stageBar.style.display = "flex";
    stageLabel.textContent = stage;
    stageDetail.textContent = detail || "";
  }

  function setBusy(state) {
    busy = state;
    sendBtn.disabled = state;
    inputEl.disabled = state;
    sendBtn.textContent = state ? "..." : "Send";
  }

  /* ── WebSocket message handler ────────────────────────────────── */

  function handleAgentMessage(data) {
    switch (data.type) {
      case "connected":
        break;

      case "text_delta":
        // Streaming: append text to current agent message
        if (!streamingEl) {
          streamingEl = createMessageEl("agent", "");
          messagesEl.appendChild(streamingEl);
        }
        const body = streamingEl.querySelector(".sd-message__body");
        body.textContent += data.text;
        scrollToBottom();
        break;

      case "message":
        // Full message — replace streaming element or create new
        if (streamingEl) {
          const msgBody = streamingEl.querySelector(".sd-message__body");
          msgBody.textContent = data.content;
          streamingEl = null;
        } else {
          messagesEl.appendChild(createMessageEl("agent", data.content));
        }
        scrollToBottom();
        break;

      case "stage":
        setStage(data.stage, data.detail);
        if (data.stage === "DONE") {
          streamingEl = null;
          setBusy(false);
        }
        break;

      case "error":
        setStage(null);
        messagesEl.appendChild(createMessageEl("system", data.message));
        streamingEl = null;
        setBusy(false);
        scrollToBottom();
        break;

      default:
        console.log("[SUPER DUPER] Unknown event:", data);
    }
  }

  function handleStatus(status) {
    statusEl.textContent = status === "connected" ? "" : status;
    statusEl.className = `sd-header__status sd-header__status--${status}`;
  }

  /* ── Connect ──────────────────────────────────────────────────── */

  const conn = new AgentConnection(handleAgentMessage, handleStatus);
  conn.connect();

  /* ── Send message ─────────────────────────────────────────────── */

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || busy) return;

    // Show user message
    messagesEl.appendChild(createMessageEl("user", text));
    inputEl.value = "";
    scrollToBottom();

    // Capture current workflow from ComfyUI canvas
    let workflow = null;
    try {
      const result = await app.graphToPrompt();
      if (result && result.output) {
        workflow = result.output;
      }
    } catch (e) {
      console.warn("[SUPER DUPER] Could not capture workflow:", e);
    }

    // Send to agent via WebSocket (include workflow if captured)
    const payload = { content: text };
    if (workflow) {
      payload.workflow = workflow;
    }
    const sent = conn.send("chat", payload);
    if (!sent) {
      messagesEl.appendChild(
        createMessageEl("system", "Not connected. Reconnecting...")
      );
      scrollToBottom();
      return;
    }

    setBusy(true);
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  setTimeout(() => inputEl.focus(), 100);
}

/* ── Register with ComfyUI ────────────────────────────────────────── */

app.extensionManager.registerSidebarTab({
  id: SIDEBAR_ID,
  icon: "pi pi-bolt",
  title: SIDEBAR_TITLE,
  tooltip: "SUPER DUPER AI Co-pilot",
  type: "custom",
  render(el) {
    buildSidebar(el);
  },
});
