import { app } from "../../../scripts/app.js";
import { renderText, createTypingIndicator, createPanel, createNodePill } from "./chat.js";
import { createDispatchCard, updateAgentStatus } from "./dispatch.js";
import { createProgressPanel, updateProgress } from "./progress.js";
import { createQuickActions, updateQuickActions } from "./actions.js";

/* ── SUPER DUPER Sidebar ─────────────────────────────────────────────
 *  Registers a sidebar tab in ComfyUI's extension manager.
 *  Chat interface with WebSocket connection to agent brain.
 * ──────────────────────────────────────────────────────────────────── */

const SIDEBAR_ID = "superduper";
const SIDEBAR_TITLE = "Super Duper";

/* ── Load fonts & stylesheet ──────────────────────────────────────── */

const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap";
document.head.appendChild(fontLink);

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

/* ── Readability Controls ─────────────────────────────────────────── */

const SVG_NS = "http://www.w3.org/2000/svg";
const FONT_STEPS = ["s", "m", "l"];
const ALIGN_OPTIONS = ["left", "center", "right"];
const MODE_OPTIONS = ["comfort", "compact"];

/** Create a tiny SVG with horizontal line rects. */
function _svgLines(lineData) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "14");
  svg.setAttribute("height", "14");
  svg.setAttribute("viewBox", "0 0 14 14");
  for (const d of lineData) {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", String(d.x));
    rect.setAttribute("y", String(d.y));
    rect.setAttribute("width", String(d.w));
    rect.setAttribute("height", "1.5");
    rect.setAttribute("rx", "0.75");
    rect.setAttribute("fill", "currentColor");
    svg.appendChild(rect);
  }
  return svg;
}

/** Alignment icon — three lines with position weight. */
function _alignIcon(align) {
  const specs = {
    left:   [{ x: 1, y: 3, w: 12 }, { x: 1, y: 6.5, w: 8 },  { x: 1, y: 10, w: 10 }],
    center: [{ x: 1, y: 3, w: 12 }, { x: 3, y: 6.5, w: 8 },  { x: 2, y: 10, w: 10 }],
    right:  [{ x: 1, y: 3, w: 12 }, { x: 5, y: 6.5, w: 8 },  { x: 3, y: 10, w: 10 }],
  };
  return _svgLines(specs[align]);
}

/** Mode icon — three lines with gap spacing. */
function _modeIcon(mode) {
  const ys = mode === "comfort" ? [2, 6.25, 10.5] : [4, 6.25, 8.5];
  return _svgLines(ys.map(y => ({ x: 2, y, w: 10 })));
}

/** Create a readbar button. */
function _readBtn(title, ariaLabel) {
  const btn = document.createElement("button");
  btn.className = "sd-readbar__btn";
  btn.title = title;
  btn.setAttribute("aria-label", ariaLabel || title);
  return btn;
}

/** Create a separator. */
function _readSep() {
  const sep = document.createElement("span");
  sep.className = "sd-readbar__sep";
  return sep;
}

/** Populate the readability bar. */
function _buildReadbar(barEl, messagesEl) {
  // Restore saved preferences (comfort + m + left are defaults)
  const savedFont  = localStorage.getItem("sd-font")  || "m";
  const savedAlign = localStorage.getItem("sd-align") || "left";
  const savedMode  = localStorage.getItem("sd-mode")  || "comfort";

  messagesEl.dataset.font  = savedFont;
  messagesEl.dataset.align = savedAlign;
  messagesEl.dataset.mode  = savedMode;

  // ── Font size: A↓ A↑ ──
  const fontDown = _readBtn("Smaller text");
  fontDown.classList.add("sd-readbar__btn--font-down");
  fontDown.textContent = "A";
  fontDown.addEventListener("click", () => {
    const i = FONT_STEPS.indexOf(messagesEl.dataset.font);
    if (i > 0) {
      messagesEl.dataset.font = FONT_STEPS[i - 1];
      localStorage.setItem("sd-font", messagesEl.dataset.font);
    }
  });
  barEl.appendChild(fontDown);

  const fontUp = _readBtn("Larger text");
  fontUp.classList.add("sd-readbar__btn--font-up");
  fontUp.textContent = "A";
  fontUp.addEventListener("click", () => {
    const i = FONT_STEPS.indexOf(messagesEl.dataset.font);
    if (i < FONT_STEPS.length - 1) {
      messagesEl.dataset.font = FONT_STEPS[i + 1];
      localStorage.setItem("sd-font", messagesEl.dataset.font);
    }
  });
  barEl.appendChild(fontUp);

  barEl.appendChild(_readSep());

  // ── Alignment: left / center / right ──
  const alignBtns = {};
  for (const align of ALIGN_OPTIONS) {
    const btn = _readBtn(`${align.charAt(0).toUpperCase() + align.slice(1)} align`);
    btn.appendChild(_alignIcon(align));
    btn.addEventListener("click", () => {
      messagesEl.dataset.align = align;
      localStorage.setItem("sd-align", align);
      _syncActive();
    });
    alignBtns[align] = btn;
    barEl.appendChild(btn);
  }

  barEl.appendChild(_readSep());

  // ── Mode: comfort / compact ──
  const modeBtns = {};
  const modeLabels = { comfort: "Comfort reading", compact: "Compact reading" };
  for (const mode of MODE_OPTIONS) {
    const btn = _readBtn(modeLabels[mode]);
    btn.appendChild(_modeIcon(mode));
    btn.addEventListener("click", () => {
      messagesEl.dataset.mode = mode;
      localStorage.setItem("sd-mode", mode);
      _syncActive();
    });
    modeBtns[mode] = btn;
    barEl.appendChild(btn);
  }

  // Sync active highlights
  function _syncActive() {
    for (const [k, b] of Object.entries(alignBtns)) {
      b.classList.toggle("sd-readbar__btn--active", k === messagesEl.dataset.align);
    }
    for (const [k, b] of Object.entries(modeBtns)) {
      b.classList.toggle("sd-readbar__btn--active", k === messagesEl.dataset.mode);
    }
  }
  _syncActive();
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
    <div class="sd-readbar" id="sd-readbar"></div>
    <div id="sd-quick-actions"></div>
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
  const readbarEl = el.querySelector("#sd-readbar");

  // Build readability controls
  _buildReadbar(readbarEl, messagesEl);

  // Quick action chips
  const quickActionsSlot = el.querySelector("#sd-quick-actions");
  const ACTION_MESSAGES = {
    run: "Run the current workflow",
    validate: "Validate the current workflow and check for issues",
    changes: "What changes have been made to the workflow?",
    undo: "Undo the last workflow change",
    optimize: "Suggest optimizations for the current workflow",
    repair: "Find and install missing nodes for this workflow",
  };
  const quickActionsEl = createQuickActions((actionId) => {
    const message = ACTION_MESSAGES[actionId];
    if (!message || busy) return;
    messagesEl.appendChild(createMessageEl("user", message));
    const sent = conn.send("action", { action: "agent_message", message });
    if (sent) {
      setBusy(true);
      _showThinking();
      scrollToBottom();
    }
  });
  quickActionsSlot.appendChild(quickActionsEl);

  let busy = false;
  let streamingEl = null;     // element receiving streamed text deltas
  let streamAccum = "";       // accumulated text during streaming
  let typingIndicator = null; // typing dots shown during stream
  let dispatchCard = null;    // current agent dispatch card element
  let progressPanel = null;   // current progress panel element
  let pipelineState = {};     // tracks pipeline stage statuses

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
        _clearThinking();
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
        _clearThinking();
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
          dispatchCard = null;
          if (progressPanel) {
            progressPanel.remove();
            progressPanel = null;
          }
          pipelineState = {};
          setBusy(false);
          // Update quick actions
          updateQuickActions({
            workflowLoaded: true,
            hasChanges: false,
            canUndo: false,
            isValid: true,
          });
        }
        break;

      case "progress":
        // ComfyUI execution progress — update bar and node detail
        if (progressPanel) {
          updateProgress(progressPanel, {
            progress: data.progress != null ? data.progress : null,
            currentNode: data.current_node || data.currentNode,
            nodeIndex: data.node_index != null ? data.node_index : data.nodeIndex,
            nodeTotal: data.node_total != null ? data.node_total : data.nodeTotal,
            etaSeconds: data.eta_seconds != null ? data.eta_seconds : data.etaSeconds,
          });
          scrollToBottom();
        }
        break;

      case "agent_dispatch": {
        _clearThinking();
        // Insert dispatch card ABOVE upcoming chat messages
        const cardData = {
          prompt: data.prompt,
          agents: data.agents || [],
          estimate: data.estimate_seconds != null ? `~${data.estimate_seconds}s` : undefined,
          nodeCount: data.node_count,
        };
        dispatchCard = createDispatchCard(cardData);
        messagesEl.appendChild(dispatchCard);

        // Create progress panel below the dispatch card
        pipelineState = { router: "waiting", intent: "waiting", execution: "waiting", verify: "waiting" };
        progressPanel = createProgressPanel();
        messagesEl.appendChild(progressPanel);

        scrollToBottom();
        break;
      }

      case "agent_status":
        if (dispatchCard) {
          updateAgentStatus(
            dispatchCard,
            data.agent_key,
            data.status,
            data.message,
            data.duration
          );
        }
        // Update pipeline level in progress panel
        if (progressPanel && data.agent_key in pipelineState) {
          pipelineState[data.agent_key] = data.status;
          updateProgress(progressPanel, { pipeline: pipelineState });
        }
        // Dispatch node touch events so canvas highlights agent-touched nodes
        if (data.nodes_touched && data.nodes_touched.length > 0) {
          const color = {
            router: "#00BB81",
            intent: "#FFD500",
            execution: "#64B5F6",
            verify: "#FF6E6E",
            doctor: "#B39DDB"
          }[data.agent_key] || "#00BB81";
          for (const nodeId of data.nodes_touched) {
            document.dispatchEvent(new CustomEvent("superduper:node_touch", {
              detail: { nodeId: nodeId, agentColor: color }
            }));
          }
        }
        scrollToBottom();
        break;

      case "error":
        _clearThinking();
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

  function _showThinking() {
    const el = document.createElement("div");
    el.className = "sd-thinking";

    const dot = document.createElement("span");
    dot.className = "sd-thinking__dot";
    el.appendChild(dot);

    const label = document.createElement("span");
    label.textContent = "Connecting to agents...";
    el.appendChild(label);

    messagesEl.appendChild(el);
    scrollToBottom();
  }

  function _clearThinking() {
    const el = messagesEl.querySelector(".sd-thinking");
    if (el) el.remove();
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || busy) return;

    // Show user message immediately
    messagesEl.appendChild(createMessageEl("user", text));
    inputEl.value = "";

    // Show thinking indicator right away
    _showThinking();
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
      _clearThinking();
      messagesEl.appendChild(
        createMessageEl("system", "Not connected. Reconnecting...")
      );
      scrollToBottom();
      return;
    }

    setBusy(true);
  }

  sendBtn.addEventListener("click", sendMessage);

  // Panel action button delegation — sends action to agent via WebSocket
  messagesEl.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn || busy) return;

    const action = btn.dataset.action;
    const payload = { action };

    if (action === "agent_message" && btn.dataset.message) {
      // Show the message as a user message in chat
      messagesEl.appendChild(createMessageEl("user", btn.dataset.message));
      payload.message = btn.dataset.message;
    } else if (action === "install_node_pack") {
      if (btn.dataset.url) payload.url = btn.dataset.url;
      if (btn.dataset.name) payload.name = btn.dataset.name;
    }

    const sent = conn.send("action", payload);
    if (sent) {
      setBusy(true);
      _showThinking();
      scrollToBottom();
    }
  });

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

          // Flash the node with a jade accent bar (visual ping)
          document.dispatchEvent(new CustomEvent("superduper:node_touch", {
            detail: { nodeId: pill.dataset.nodeId, agentColor: "#00BB81" }
          }));
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

  // ── Standalone execution tracking (Queue Prompt without agent) ────
  let standaloneExecution = false;

  document.addEventListener("superduper:execution_start", () => {
    // Only create a standalone progress panel if agent isn't driving
    if (progressPanel) return;
    standaloneExecution = true;
    progressPanel = createProgressPanel();
    messagesEl.appendChild(progressPanel);
    updateProgress(progressPanel, { progress: null });
    scrollToBottom();
  });

  document.addEventListener("superduper:execution_progress", (e) => {
    if (!progressPanel) return;
    updateProgress(progressPanel, {
      currentNode: e.detail.nodeName,
      nodeIndex: e.detail.nodeIndex,
    });
    scrollToBottom();
  });

  document.addEventListener("superduper:node_progress", (e) => {
    if (!progressPanel) return;
    updateProgress(progressPanel, { progress: e.detail.progress });
  });

  document.addEventListener("superduper:execution_complete", () => {
    if (progressPanel && standaloneExecution) {
      updateProgress(progressPanel, { progress: 1 });
      // Fade out after a brief moment
      setTimeout(() => {
        if (progressPanel && standaloneExecution) {
          progressPanel.remove();
          progressPanel = null;
          standaloneExecution = false;
        }
      }, 2000);
    }
  });

  // Update progress panel with current node name (agent-driven or standalone)
  document.addEventListener("superduper:node_executing", (e) => {
    if (!progressPanel) return;
    updateProgress(progressPanel, { currentNode: e.detail.nodeName });
  });

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
