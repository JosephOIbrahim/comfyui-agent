/**
 * Quick Action Chips — context-aware suggestion buttons.
 *
 * Appears below the input bar with contextual actions based on
 * the current workflow and agent state.
 *
 * Part of the v2.0 Pentagram-caliber UI redesign.
 */

let _containerEl = null;
let _onAction = null;

const ACTIONS = {
  run: { label: "Run workflow", icon: "\u25B6" },
  repair: { label: "Repair", icon: "\u2692" },
  changes: { label: "What changed?", icon: "\u0394" },
  undo: { label: "Undo", icon: "\u21A9" },
  optimize: { label: "Optimize", icon: "\u26A1" },
  validate: { label: "Validate", icon: "\u2713" },
};

/**
 * Create the quick actions container.
 * @param {Function} onAction - callback(actionId) when a chip is clicked
 */
export function createQuickActions(onAction) {
  _onAction = onAction;
  _containerEl = document.createElement("div");
  _containerEl.className = "sd-quick-actions";
  _containerEl.style.display = "none";
  return _containerEl;
}

/**
 * Update which actions are visible based on state.
 * @param {Object} state - { workflowLoaded, hasChanges, canUndo, isValid }
 */
export function updateQuickActions(state) {
  if (!_containerEl) return;
  _containerEl.innerHTML = "";

  const visible = [];

  if (state.workflowLoaded && state.isValid !== false) {
    visible.push("run");
  }
  if (state.workflowLoaded) {
    visible.push("validate");
  }
  if (state.hasChanges) {
    visible.push("changes");
  }
  if (state.canUndo) {
    visible.push("undo");
  }
  if (state.workflowLoaded) {
    visible.push("optimize");
  }

  if (visible.length === 0) {
    _containerEl.style.display = "none";
    return;
  }

  _containerEl.style.display = "flex";

  for (const id of visible) {
    const def = ACTIONS[id];
    if (!def) continue;

    const chip = document.createElement("button");
    chip.className = "sd-quick-action";
    chip.textContent = def.label;
    chip.addEventListener("click", () => {
      if (_onAction) _onAction(id);
    });
    _containerEl.appendChild(chip);
  }
}
