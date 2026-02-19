/* ── SUPER DUPER Chat Renderer ──────────────────────────────────────
 *  Text rendering (Phase 1) + Panel component factories (Phase 2).
 *  No innerHTML anywhere — all DOM built via createElement + textContent.
 * ──────────────────────────────────────────────────────────────────── */

// ── Slot Colors ────────────────────────────────────────────────────

export const SLOT_COLORS = {
  clip:         { color: "#FFD500", varName: "--sd-slot-clip" },
  clip_vision:  { color: "#A8DADC", varName: "--sd-slot-clip-vision" },
  conditioning: { color: "#FFA931", varName: "--sd-slot-conditioning" },
  controlnet:   { color: "#6EE7B7", varName: "--sd-slot-controlnet" },
  image:        { color: "#64B5F6", varName: "--sd-slot-image" },
  latent:       { color: "#FF9CF9", varName: "--sd-slot-latent" },
  mask:         { color: "#81C784", varName: "--sd-slot-mask" },
  model:        { color: "#B39DDB", varName: "--sd-slot-model" },
  style_model:  { color: "#C2FFAE", varName: "--sd-slot-style-model" },
  vae:          { color: "#FF6E6E", varName: "--sd-slot-vae" },
  noise:        { color: "#B0B0B0", varName: "--sd-slot-noise" },
  guider:       { color: "#66FFFF", varName: "--sd-slot-guider" },
  sampler:      { color: "#ECB4B4", varName: "--sd-slot-sampler" },
  sigmas:       { color: "#CDFFCD", varName: "--sd-slot-sigmas" },
};

function _slotColor(slotType) {
  const s = SLOT_COLORS[slotType];
  return s ? s.color : "#9a9caa";
}

// ── Text Rendering ─────────────────────────────────────────────────

/**
 * Parse raw text into a DocumentFragment with styled elements.
 * Handles: ```code blocks```, `inline code`, **bold**, paragraph breaks.
 * NO innerHTML — all content via textContent.
 */
export function renderText(rawText) {
  const frag = document.createDocumentFragment();
  if (!rawText) return frag;

  // Split on triple-backtick code blocks
  const parts = rawText.split(/(```[\s\S]*?```)/);

  for (const part of parts) {
    if (part.startsWith("```") && part.endsWith("```")) {
      // Code block — strip fences and optional language tag
      let code = part.slice(3, -3);
      // Remove optional language identifier on first line
      const nlIdx = code.indexOf("\n");
      if (nlIdx !== -1 && nlIdx < 20 && /^[a-zA-Z]*$/.test(code.slice(0, nlIdx).trim())) {
        code = code.slice(nlIdx + 1);
      }
      // Trim leading/trailing newline
      if (code.startsWith("\n")) code = code.slice(1);
      if (code.endsWith("\n")) code = code.slice(0, -1);

      const pre = document.createElement("pre");
      pre.className = "sd-code-block";
      pre.textContent = code;
      frag.appendChild(pre);
    } else {
      // Text content — split into paragraphs on double newlines
      const paragraphs = part.split(/\n{2,}/);
      for (const para of paragraphs) {
        const trimmed = para.trim();
        if (!trimmed) continue;

        const p = document.createElement("p");
        _appendInlineMarkup(p, trimmed);
        frag.appendChild(p);
      }
    }
  }

  return frag;
}

/**
 * Parse inline markup (**bold** and `code`) and append nodes to parent.
 */
function _appendInlineMarkup(parent, text) {
  // Tokenize on **bold** and `code` patterns
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/);

  for (const token of tokens) {
    if (token.startsWith("**") && token.endsWith("**")) {
      // Bold
      const strong = document.createElement("strong");
      strong.className = "sd-text-bold";
      strong.textContent = token.slice(2, -2);
      parent.appendChild(strong);
    } else if (token.startsWith("`") && token.endsWith("`")) {
      // Inline code
      const code = document.createElement("code");
      code.className = "sd-code-inline";
      code.textContent = token.slice(1, -1);
      parent.appendChild(code);
    } else if (token) {
      _appendInlineText(parent, token);
    }
  }
}

/**
 * Append plain text, converting single newlines to <br>.
 */
function _appendInlineText(parent, text) {
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (i > 0) parent.appendChild(document.createElement("br"));
    if (lines[i]) parent.appendChild(document.createTextNode(lines[i]));
  }
}

// ── Typing Indicator ───────────────────────────────────────────────

export function createTypingIndicator() {
  const el = document.createElement("span");
  el.className = "sd-typing";
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("span");
    dot.className = "sd-typing__dot";
    el.appendChild(dot);
  }
  return el;
}

// ── Panel Component Factories ──────────────────────────────────────

/**
 * Create a colored slot tag chip.
 * @param {string} label - Node/concept name
 * @param {string} slotType - Slot type key (e.g., "model", "vae")
 */
export function createSlotTag(label, slotType) {
  const color = _slotColor(slotType);
  const tag = document.createElement("span");
  tag.className = "sd-slot-tag";
  tag.style.background = color + "1f"; // 12% opacity
  tag.style.border = `1px solid ${color}40`; // 25% opacity
  tag.style.color = color;

  const dot = document.createElement("span");
  dot.className = "sd-slot-tag__dot";
  dot.style.background = color;
  dot.style.boxShadow = `0 0 4px ${color}8c`; // 55% glow
  tag.appendChild(dot);

  const txt = document.createTextNode(label);
  tag.appendChild(txt);

  return tag;
}

/**
 * Create a collapsible section.
 * @param {string} title
 * @param {string} dotColor - CSS color string
 * @param {number|string} count - Badge count
 * @param {boolean} defaultOpen
 * @param {Function} buildBody - () => HTMLElement (lazy, called on first expand)
 */
export function createCollapsibleSection(title, dotColor, count, defaultOpen, buildBody) {
  const section = document.createElement("div");
  section.className = "sd-section" + (defaultOpen ? " sd-section--open" : "");

  // Trigger button
  const trigger = document.createElement("button");
  trigger.className = "sd-section__trigger";

  const dot = document.createElement("span");
  dot.className = "sd-section__dot";
  dot.style.background = dotColor;
  dot.style.boxShadow = `0 0 4px ${dotColor}8c`;
  trigger.appendChild(dot);

  const titleEl = document.createElement("span");
  titleEl.className = "sd-section__title";
  titleEl.textContent = title;
  trigger.appendChild(titleEl);

  if (count != null) {
    const countEl = document.createElement("span");
    countEl.className = "sd-section__count";
    countEl.textContent = String(count);
    trigger.appendChild(countEl);
  }

  const chevron = document.createElement("span");
  chevron.className = "sd-section__chevron";
  chevron.textContent = "\u25b8"; // right-pointing triangle
  trigger.appendChild(chevron);

  section.appendChild(trigger);

  // Body (lazy)
  const body = document.createElement("div");
  body.className = "sd-section__body";
  let built = defaultOpen;
  if (defaultOpen && buildBody) {
    const content = buildBody();
    if (content) body.appendChild(content);
  }
  section.appendChild(body);

  // Toggle
  trigger.addEventListener("click", () => {
    const isOpen = section.classList.toggle("sd-section--open");
    if (isOpen && !built && buildBody) {
      built = true;
      const content = buildBody();
      if (content) body.appendChild(content);
    }
  });

  return section;
}

/**
 * Create a flow chain showing signal path through nodes.
 * @param {Array<{label: string, slotType: string}>} nodes
 * @param {string} [pathLabel] - Optional path label prefix
 */
export function createFlowChain(nodes, pathLabel) {
  const chain = document.createElement("div");
  chain.className = "sd-flow-chain";

  if (pathLabel) {
    const lbl = document.createElement("span");
    lbl.className = "sd-flow-chain__label";
    lbl.textContent = pathLabel;
    chain.appendChild(lbl);
  }

  for (let i = 0; i < nodes.length; i++) {
    if (i > 0) {
      const arrow = document.createElement("span");
      arrow.className = "sd-flow-chain__arrow";
      arrow.textContent = "\u2192"; // →
      chain.appendChild(arrow);
    }

    const node = document.createElement("span");
    node.className = "sd-flow-chain__node";
    const color = _slotColor(nodes[i].slotType);
    node.style.color = color;
    node.style.background = color + "1a"; // ~10% opacity
    node.textContent = nodes[i].label;
    chain.appendChild(node);
  }

  return chain;
}

/**
 * Create a key-value detail row with dotted leader.
 * @param {string} label
 * @param {string} value
 */
export function createDetailRow(label, value) {
  const row = document.createElement("div");
  row.className = "sd-detail-row";

  const lbl = document.createElement("span");
  lbl.className = "sd-detail-row__label";
  lbl.textContent = label;
  row.appendChild(lbl);

  const leader = document.createElement("span");
  leader.className = "sd-detail-row__leader";
  row.appendChild(leader);

  const val = document.createElement("span");
  val.className = "sd-detail-row__value";
  val.textContent = value;
  row.appendChild(val);

  return row;
}

/**
 * Create a capability row.
 * @param {string} label
 * @param {string} desc
 * @param {string} [color] - Dot color, defaults to brand
 */
export function createCapabilityRow(label, desc, color) {
  const row = document.createElement("div");
  row.className = "sd-capability-row";

  const dot = document.createElement("span");
  dot.className = "sd-capability-row__dot";
  dot.style.background = color || "var(--sd-brand)";
  row.appendChild(dot);

  const lbl = document.createElement("span");
  lbl.className = "sd-capability-row__label";
  lbl.textContent = label;
  row.appendChild(lbl);

  const sep = document.createElement("span");
  sep.className = "sd-capability-row__sep";
  row.appendChild(sep);

  const d = document.createElement("span");
  d.className = "sd-capability-row__desc";
  d.textContent = desc;
  row.appendChild(d);

  return row;
}

/**
 * Create an action footer.
 * @param {string} status - Status text
 * @param {Array<{label: string, variant: string}>} actions - Buttons
 */
export function createActionFooter(status, actions) {
  const footer = document.createElement("div");
  footer.className = "sd-panel__footer";

  const statusEl = document.createElement("span");
  statusEl.className = "sd-panel__footer-status";
  statusEl.textContent = status || "";
  footer.appendChild(statusEl);

  if (actions && actions.length) {
    const actionsEl = document.createElement("div");
    actionsEl.className = "sd-panel__footer-actions";
    for (const action of actions) {
      const btn = document.createElement("button");
      btn.className = `sd-btn sd-btn--${action.variant || "secondary"}`;
      btn.textContent = action.label;
      actionsEl.appendChild(btn);
    }
    footer.appendChild(actionsEl);
  }

  return footer;
}

/**
 * Create a full panel from panelData.
 * @param {Object} panelData - Panel schema from backend
 * @returns {HTMLElement|null}
 *
 * panelData schema:
 * {
 *   type: "workflow_analysis"|"discovery"|"execution"|"error",
 *   header: { label, badge, title, summary, stats: [{value,label}] },
 *   sections: [{
 *     title, dotColor, count, defaultOpen,
 *     type: "flow_chain"|"slot_tags"|"capability_rows"|"detail_rows",
 *     data: { ... type-specific ... }
 *   }],
 *   footer: { status, actions: [{label,variant}] }
 * }
 */
export function createPanel(panelData) {
  if (!panelData) return null;

  const panel = document.createElement("div");
  panel.className = "sd-panel";

  // Header
  if (panelData.header) {
    const header = document.createElement("div");
    header.className = "sd-panel__header";

    // Label row
    const labelRow = document.createElement("div");
    labelRow.className = "sd-panel__label-row";

    const label = document.createElement("span");
    label.className = "sd-panel__label";
    label.textContent = panelData.header.label || "";
    labelRow.appendChild(label);

    if (panelData.header.badge) {
      const badge = document.createElement("span");
      badge.className = "sd-panel__badge";
      badge.textContent = panelData.header.badge;
      labelRow.appendChild(badge);
    }
    header.appendChild(labelRow);

    // Title
    if (panelData.header.title) {
      const title = document.createElement("div");
      title.className = "sd-panel__title";
      title.textContent = panelData.header.title;
      header.appendChild(title);
    }

    // Summary
    if (panelData.header.summary) {
      const summary = document.createElement("div");
      summary.className = "sd-panel__summary";
      summary.textContent = panelData.header.summary;
      header.appendChild(summary);
    }

    // Stats
    if (panelData.header.stats && panelData.header.stats.length) {
      const stats = document.createElement("div");
      stats.className = "sd-panel__stats";
      for (const s of panelData.header.stats) {
        const stat = document.createElement("div");
        stat.className = "sd-panel__stat";

        const val = document.createElement("span");
        val.className = "sd-panel__stat-value";
        val.textContent = String(s.value);
        stat.appendChild(val);

        const lbl = document.createElement("span");
        lbl.className = "sd-panel__stat-label";
        lbl.textContent = s.label;
        stat.appendChild(lbl);

        stats.appendChild(stat);
      }
      header.appendChild(stats);
    }

    panel.appendChild(header);
  }

  // Sections
  if (panelData.sections) {
    for (const sec of panelData.sections) {
      const dotColor = sec.dotColor || "var(--sd-brand)";
      const section = createCollapsibleSection(
        sec.title,
        dotColor,
        sec.count,
        sec.defaultOpen || false,
        () => _buildSectionBody(sec)
      );
      panel.appendChild(section);
    }
  }

  // Footer
  if (panelData.footer) {
    const footer = createActionFooter(
      panelData.footer.status,
      panelData.footer.actions
    );
    panel.appendChild(footer);
  }

  return panel;
}

/**
 * Build the body content for a section based on its type.
 */
function _buildSectionBody(sec) {
  const container = document.createDocumentFragment();

  switch (sec.type) {
    case "flow_chain": {
      const chains = sec.data.chains || [sec.data];
      for (const chainData of chains) {
        const nodes = (chainData.nodes || []).map(n =>
          typeof n === "string" ? { label: n, slotType: "model" } : n
        );
        container.appendChild(createFlowChain(nodes, chainData.label));
      }
      break;
    }

    case "slot_tags": {
      const wrap = document.createElement("div");
      wrap.className = "sd-slot-tags";
      for (const tag of (sec.data.tags || [])) {
        wrap.appendChild(createSlotTag(tag.label, tag.slotType));
      }
      container.appendChild(wrap);
      break;
    }

    case "detail_rows": {
      for (const row of (sec.data.rows || [])) {
        container.appendChild(createDetailRow(row.label, row.value));
      }
      break;
    }

    case "capability_rows": {
      for (const row of (sec.data.rows || [])) {
        container.appendChild(createCapabilityRow(row.label, row.desc, row.color));
      }
      break;
    }

    default: {
      // Fallback: render as text
      if (sec.data && sec.data.text) {
        const p = document.createElement("p");
        p.className = "sd-text-body";
        p.textContent = sec.data.text;
        container.appendChild(p);
      }
    }
  }

  return container;
}
