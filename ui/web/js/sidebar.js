import { app } from "../../../scripts/app.js";
import { renderText, createTypingIndicator, createPanel, createNodePill } from "./chat.js";

/* ── SUPER DUPER Sidebar ─────────────────────────────────────────────
 *  Registers a sidebar tab in ComfyUI's extension manager.
 *  Chat interface with WebSocket connection to agent brain.
 * ──────────────────────────────────────────────────────────────────── */

const SIDEBAR_ID = "superduper";
const SIDEBAR_TITLE = "Super Duper";

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

  const body = document.createElement("div");
  body.className = "sd-message__body";

  if (role === "agent") {
    // Rich text rendering for agent messages
    body.classList.add("sd-text-body");
    body.appendChild(renderText(text));
  } else {
    // Plain text for user and system messages
    body.textContent = text;
  }

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

  el.innerHTML = `
    <div class="sd-header">
      <span class="sd-header__title">${SIDEBAR_TITLE}</span>
      <span class="sd-header__status" id="sd-status"></span>
    </div>
    <div class="sd-stage-bar" id="sd-stage" style="display:none">
      <span class="sd-stage__label" id="sd-stage-label"></span>
      <span class="sd-stage__detail" id="sd-stage-detail"></span>
    </div>
    <div class="sd-messages" id="sd-messages" role="log" aria-live="polite" aria-label="Chat with Super Duper AI">
      <div class="sd-message sd-message--system">
        <span class="sd-message__body">What would you like to do with your workflow?</span>
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
      <button id="sd-send" class="sd-send-btn" aria-label="Send message">Send</button>
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
  let streamingEl = null;     // element receiving streamed text deltas
  let streamAccum = "";       // accumulated text during streaming
  let typingIndicator = null; // typing dots shown during stream

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setStage(stage, detail, nodesTouched) {
    if (stage === "DONE" || !stage) {
      stageBar.style.display = "none";
      return;
    }
    stageBar.style.display = "flex";
    stageLabel.textContent = stage;
    stageDetail.textContent = detail || "";

    // Remove existing pills
    const existingPills = stageBar.querySelectorAll(".sd-node-pill");
    existingPills.forEach(p => p.remove());

    // Render node pills if provided
    if (nodesTouched && nodesTouched.length > 0) {
      for (const node of nodesTouched) {
        const pill = createNodePill(
          node.class_type,
          node.slot_type,
          node.node_id
        );
        stageBar.appendChild(pill);
      }
    }
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
        // Streaming: show typing indicator, accumulate text
        if (!streamingEl) {
          streamingEl = document.createElement("div");
          streamingEl.className = "sd-message sd-message--agent";

          const label = document.createElement("span");
          label.className = "sd-message__label";
          label.textContent = "Agent";
          streamingEl.appendChild(label);

          const body = document.createElement("div");
          body.className = "sd-message__body sd-text-body";
          streamingEl.appendChild(body);

          messagesEl.appendChild(streamingEl);
          streamAccum = "";
        }

        streamAccum += data.text;

        // Show typing indicator during streaming
        const streamBody = streamingEl.querySelector(".sd-message__body");
        if (!typingIndicator) {
          typingIndicator = createTypingIndicator();
          streamBody.appendChild(typingIndicator);
        }

        scrollToBottom();
        break;

      case "message": {
        // Full message — render with rich text, replacing streaming element
        const finalText = data.content || streamAccum;

        if (streamingEl) {
          // Replace streaming content with final rendered text
          const body = streamingEl.querySelector(".sd-message__body");
          body.textContent = ""; // clear typing indicator
          body.classList.add("sd-text-body");
          body.appendChild(renderText(finalText));
          streamingEl = null;
          streamAccum = "";
          typingIndicator = null;
        } else {
          messagesEl.appendChild(createMessageEl("agent", finalText));
        }
        scrollToBottom();
        break;
      }

      case "panel": {
        // Structured panel from tool result
        const panelEl = createPanel(data.panel);
        if (panelEl) {
          // Wrap in a message container for consistent spacing
          const wrapper = document.createElement("div");
          wrapper.className = "sd-message sd-message--agent";
          wrapper.style.padding = "0";
          wrapper.style.background = "none";
          wrapper.appendChild(panelEl);
          messagesEl.appendChild(wrapper);
          scrollToBottom();
        }
        break;
      }

      case "stage":
        setStage(data.stage, data.detail, data.nodes_touched);

        // Show node interaction indicator in chat
        if (data.nodes_touched && data.nodes_touched.length > 0 && data.stage !== "DONE") {
          const nodeMsg = document.createElement("div");
          nodeMsg.className = "sd-message sd-message--node-activity";

          const activityLabel = document.createElement("span");
          activityLabel.className = "sd-node-activity__label";
          activityLabel.textContent = data.stage === "PILOT" ? "Modifying" : "Inspecting";
          nodeMsg.appendChild(activityLabel);

          for (const node of data.nodes_touched) {
            const pill = createNodePill(
              node.class_type,
              node.slot_type,
              node.node_id
            );
            nodeMsg.appendChild(pill);
          }

          messagesEl.appendChild(nodeMsg);
          scrollToBottom();
        }

        if (data.stage === "DONE") {
          // Clean up streaming state
          if (streamingEl && streamAccum) {
            const body = streamingEl.querySelector(".sd-message__body");
            body.textContent = "";
            body.classList.add("sd-text-body");
            body.appendChild(renderText(streamAccum));
          }
          streamingEl = null;
          streamAccum = "";
          typingIndicator = null;
          setBusy(false);
        }
        break;

      case "error":
        setStage(null);
        messagesEl.appendChild(createMessageEl("system", data.message));
        streamingEl = null;
        streamAccum = "";
        typingIndicator = null;
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

  // Node pill canvas interaction (event delegation)
  messagesEl.addEventListener("click", (e) => {
    const pill = e.target.closest(".sd-node-pill");
    if (!pill || !pill.dataset.nodeId) return;

    try {
      const graph = app.graph;
      if (graph) {
        const node = graph.getNodeById(parseInt(pill.dataset.nodeId, 10));
        if (node) {
          app.canvas.deselectAllNodes();
          app.canvas.selectNode(node);
          app.canvas.centerOnNode(node);
        }
      }
    } catch (err) {
      console.warn("[Super Duper] Could not highlight node:", err);
    }
  });

  // Node pill hover highlighting
  messagesEl.addEventListener("mouseenter", (e) => {
    const pill = e.target.closest(".sd-node-pill");
    if (!pill || !pill.dataset.nodeId) return;

    try {
      const graph = app.graph;
      if (graph) {
        const node = graph.getNodeById(parseInt(pill.dataset.nodeId, 10));
        if (node) {
          node._sd_orig_color = node.color;
          const color = getComputedStyle(pill).getPropertyValue("--pill-color").trim();
          node.color = color;
          app.canvas.setDirty(true, true);
        }
      }
    } catch (err) { /* silent */ }
  }, true);

  messagesEl.addEventListener("mouseleave", (e) => {
    const pill = e.target.closest(".sd-node-pill");
    if (!pill || !pill.dataset.nodeId) return;

    try {
      const graph = app.graph;
      if (graph) {
        const node = graph.getNodeById(parseInt(pill.dataset.nodeId, 10));
        if (node && node._sd_orig_color !== undefined) {
          node.color = node._sd_orig_color;
          delete node._sd_orig_color;
          app.canvas.setDirty(true, true);
        }
      }
    } catch (err) { /* silent */ }
  }, true);

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
  tooltip: "Super Duper AI Co-pilot",
  type: "custom",
  render(el) {
    buildSidebar(el);
  },
});
