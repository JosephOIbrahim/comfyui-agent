# ARCH-1 — Outside-In vs Inside-Out for Comfy-Cozy

**Type:** MOE Council Brief — DECISION ONLY, NO CODE
**Status:** PENDING council session
**Owner:** Joe Ibrahim (decision authority)
**Roles in council:** SCOUT (current state), ARCHITECT (target state), CRUCIBLE (adversarial), FORGE (cost estimate)

---

## 0. The Question

**Should Comfy-Cozy migrate from its current outside-in architecture (standalone process talking to ComfyUI on `localhost:8188`) to an inside-out architecture (Comfy-Cozy as a ComfyUI custom node package, Moneta consumed in-process)?**

Three possible council outcomes:

- 🟢 **GREEN** — Approve full inside-out migration; phase 2 implementation plan follows
- 🟡 **YELLOW** — Approve hybrid (some inside-out surfaces, outside-in transport retained for non-cognitive paths)
- 🔴 **RED** — Reject; outside-in stays; ARCH-5 dies as a consequence

---

## 1. Background

### 1.1 Current architecture (outside-in)

```
   Joe writes a prompt
        ↓
   Comfy-Cozy process (standalone)
        ↓ HTTP/WebSocket
   localhost:8188
        ↓
   ComfyUI process (separate)
```

- Comfy-Cozy is a standalone Python process
- Talks to ComfyUI via REST + WebSocket
- Moneta runs adjacent to Comfy-Cozy, accessed via the comfy-moneta-bridge (HTTP transport per v0.1.0)
- ComfyUI is unchanged — Comfy-Cozy is a client, not an extension

### 1.2 Target architecture (inside-out)

```
   Joe writes a prompt
        ↓
   ComfyUI process
        ├── ComfyUI's native execution engine
        ├── Comfy-Cozy custom node package (in-process)
        │    └── Moneta SDK (in-process, zero-latency)
        └── Workflow graph (shared substrate)
```

- Comfy-Cozy installs as a custom node package in `G:\COMFY\ComfyUI\custom_nodes\`
- Moneta runs in ComfyUI's Python interpreter
- The comfy-moneta-bridge becomes a thin in-process SDK layer (or merges into Moneta directly)
- The workflow graph is the shared substrate — Comfy-Cozy authors workflow nodes, Joe authors workflow nodes, ComfyUI's engine resolves the merged result

### 1.3 The Synapse precedent

The Synapse v1.1 spec implements this exact pattern for Houdini:

> *The AI does not "use" Houdini. The AI writes to a USD stage. The artist writes to a USD stage. Houdini is simply the impartial viewport that renders their shared reality.*

Comfy-Cozy inside-out applies the same principle: ComfyUI is the impartial viewport rendering a workflow graph that both Joe and Moneta write to.

---

## 2. The Case for Inside-Out (🟢)

### 2.1 Architectural fit

Moneta was designed for in-process zero-latency consumption. The user memory line `usd-core: Python USD for in-process cognitive substrate at zero latency` is the inside-out architecture latent in the original spec. **The current HTTP transport in comfy-moneta-bridge is a shipping convenience, not an architectural intent.**

Inside-out makes the architecture match the original design.

### 2.2 Patent surface expansion

ARCH-5 (workflow graph as LIVRPS substrate) is patent-extension territory — a CIP filing on the existing claims. **ARCH-5 is only viable if ARCH-1 lands on inside-out**, because outside-in cannot author workflow nodes coherently — it can only submit prompts and receive outputs.

If patent moat expansion matters, inside-out is the precondition.

### 2.3 Cross-product validation

Validating substrate isomorphism on a *second* consumer surface (ComfyUI in addition to Houdini) strengthens the cross-product patent thesis. Two products implementing the same architectural principle on different DCC substrates is a stronger claim than one.

### 2.4 Frame B / Frame F unlock

Two capabilities only achievable inside-out:

- **Cozy Shadow Graph** (Frame F) — proposed workflow nodes ghosted in ComfyUI's editor, Joe commits with a gesture. Outside-in cannot do this; ComfyUI's frontend won't display ghosted nodes from a remote process.
- **Prepared-decision nodes** (Frame B) — Moneta-aware custom nodes that read pre-computed assertions at cook time. Outside-in cannot do this; the cook-time hook is in-process only.

### 2.5 Latency

Outside-in cognitive state access pays HTTP serialization cost on every read. Inside-out is direct Python call. For the autoresearch loop (Path B, vision evaluator gated), latency compounds — every iteration includes cognitive substrate reads.

### 2.6 Zero-latency was the design intent

From userMemories: *"usd-core: Python USD for in-process cognitive substrate at zero latency."* The current architecture does not achieve this. Inside-out does.

---

## 3. The Case for Outside-In (🔴)

### 3.1 Already shipping

comfy-moneta-bridge v0.1.0 shipped 4.29. 49 tests green. Three-repo substrate launch story standing publicly. Migration cost is real.

### 3.2 Process isolation

Outside-in keeps Comfy-Cozy and ComfyUI in separate processes. ComfyUI crashes (and it does, especially on heavy models) do not take Comfy-Cozy down. Restart isolation is a real operational property.

### 3.3 ComfyUI dependency conflict surface

ComfyUI's Python environment has aggressive pinning (PyTorch versions, transformers, custom node demands). Moneta's `usd-core` dependency may conflict with other custom nodes' demands. **This is not theoretical** — many ComfyUI custom node packages are mutually exclusive due to dependency conflicts.

### 3.4 Versioning discipline overhead

Custom node packaging requires:
- Compatibility matrix with ComfyUI versions
- Install-path discipline
- Frontend extension versioning
- Coordination with other custom nodes' frontend extensions

This is real ongoing maintenance burden.

### 3.5 Test surface re-validation

The 49 green tests in comfy-moneta-bridge currently verify HTTP transport. Inside-out reshapes the bridge — those tests need to be re-evaluated, replaced, or deleted. Some test surface area will be lost during migration.

### 3.6 Single point of failure

In-process means a bug in Moneta can crash ComfyUI. Out-of-process means it can't. This matters for production reliability.

---

## 4. The Hybrid Case (🟡)

A middle path: keep the standalone process for the autonomous-agent runtime (the prompt → workflow → execute → score → iterate loop), and add a thin inside-out surface for Frame B / Frame F (the in-editor Cozy Shadow Graph and prepared-decision nodes).

```
   Comfy-Cozy standalone process
        ↓ HTTP (preserved)
   localhost:8188
        ↑
   ComfyUI process
        └── Cozy thin custom node (NEW)
             └── Moneta SDK (in-process for Frame B/F only)
```

**What this preserves:** existing comfy-moneta-bridge, autonomous loop, process isolation, current test surface.

**What this adds:** custom node package for in-editor surfaces (Cozy Shadow Graph, prepared-decision nodes), Moneta runs in *both* processes (some duplication).

**The tradeoff:** Moneta running in two places creates state-synchronization complexity. The autonomous loop's Moneta instance and the in-editor Moneta instance see different state unless explicitly synced. **This recreates the same disconnected-experience-stores problem already flagged in ARCH-2.**

Hybrid is real but costly. The cost is paid in state coherence, not in lines of code.

---

## 5. Trade-off Matrix

| Dimension | 🔴 Outside-In | 🟡 Hybrid | 🟢 Inside-Out |
|---|---|---|---|
| Migration cost (effort) | $0 | $$ | $$$ |
| Architectural fit | Low | Medium | High |
| Patent surface | Current claims | Current + partial CIP | Full CIP path |
| Crash isolation | High | Medium | Low |
| Cognitive latency | High (HTTP) | Mixed | Zero (in-process) |
| Frame B / F achievable | No | Partial | Yes |
| ComfyUI dep conflict risk | None | Medium | High |
| Long-term maintenance | High (transport drift) | Highest (two paths) | Medium |
| Time-to-Tier-1-2-completion | Fastest | Slow | Slowest |
| State coherence | Single source | Two sources (problem) | Single source |

---

## 6. CRUCIBLE — Adversarial Pressure

The CRUCIBLE role's job is to attack the recommendation. Three adversarial framings:

### 6.1 "Inside-out is solving a problem you don't have"

Counter: Comfy-Cozy is fully autonomous. The user does not interact with the workflow graph during a Comfy-Cozy run. Frame F (Shadow Graph) and Frame B (prepared decisions) are co-pilot frames. **An autonomous agent has no co-pilot frames.**

If Comfy-Cozy stays autonomous, inside-out delivers only the latency win — and even that is dwarfed by ComfyUI rendering time, which is the actual bottleneck.

**Counter-counter:** This argument assumes Comfy-Cozy stays purely autonomous. The Phase 7 SuperDuper Panel is already a ComfyUI extension — that piece is already inside-out by design. The roadmap *is* moving toward in-editor surfaces. Inside-out gets ahead of that trajectory. **And:** even within an autonomous loop, prepared-decision nodes (Frame B) accelerate iteration speed.

### 6.2 "The dependency conflict surface kills it"

Counter: Real concern. `usd-core` is not a small dependency. ComfyUI's environment has many custom nodes with conflicting demands.

**Mitigation paths:**
- Vendor `usd-core` into Moneta (size cost)
- Narrow Moneta's deps to a minimal in-process subset
- Use ComfyUI's `--cpu` mode dependency profile as a baseline test
- Document a "Comfy-Cozy installation requires these custom nodes to be removed" exclusion list

**The honest answer:** until the scout pass (`01_SCOUT_INSIDE_OUT_v0_1.md`) runs and quantifies the conflict surface, this risk is unsized. Council should approve scout pass execution as a precondition to full inside-out commitment.

### 6.3 "You just shipped the bridge — this is avoidance"

Counter: This is the strongest adversarial framing. Three framework edits in a day = avoidance, by Joe's own rule. Is this revisiting a freshly-shipped decision?

**Honest read:** The 4.29 ship was the right move *for that day's milestone* (substrate launch). The inside-out question was always going to come up — it's latent in Moneta's design. But the *timing* matters: revisiting v0.1.0's transport architecture immediately after shipping looks like avoidance.

**Mitigation:** Council can approve inside-out as a *Phase 2 destination*, not a Phase 1.5 immediate pivot. Tier 1-2 hardening completes on outside-in. Inside-out lands after Tier-2 closeout, when the migration cost can be paid against a stable baseline.

---

## 7. Recommendation

**🟢 GREEN with timing constraint.**

Conditions:

1. Inside-out is the architectural destination
2. Tier 1-2 hardening completes on outside-in (do not interrupt momentum on the hardening track)
3. Scout pass (`01_SCOUT_INSIDE_OUT_v0_1.md`) executes immediately after Tier-2 closeout
4. Phase 2 inside-out implementation plan is drafted from scout findings, with HARD risks blocking until mitigated
5. ARCH-5 council decision (workflow LIVRPS) lands GREEN as a consequence, with patent CIP filing scheduled in the inside-out implementation phase

If any HARD risk is discovered in scout, council reconvenes for re-decision.

**🟡 Acceptable fallback** if 🟢 cannot be approved: hybrid path with *explicit acknowledgment* that state-coherence problem will need ARCH-2 resolution before any in-editor surface ships.

**🔴 Rejection consequence:** ARCH-5 dies. Patent CIP path closes. Frame B / F capabilities unbuilt. Comfy-Cozy stays a transport-coupled client of ComfyUI permanently.

---

## 8. What approval unlocks

If 🟢:

- ARCH-5 council decision becomes viable
- Scout pass execution becomes scheduled
- Phase 2 implementation roadmap drafting begins post-Tier-2
- Patent CIP filing moves into legal queue
- comfy-moneta-bridge repo gets a future deprecation timeline (concepts survive in Moneta SDK)

If 🟡:

- Limited Phase 2 work for in-editor surfaces only
- Hybrid state-coherence problem becomes a tracked tech debt item
- ARCH-5 decision deferred or partially approved

If 🔴:

- ARCH-5 closes
- Tier 1-2 + Tier 3-4 continue on current architecture
- Phase 7 SuperDuper Panel reframes as a thin frontend-only extension

---

## 9. What approval does NOT do

- Does not commit to a delivery date
- Does not approve any code changes
- Does not authorize the scout pass to begin (that's a separate trigger after Tier-2 closeout)
- Does not bind the patent CIP filing decision (that's downstream of scout findings)

---

## 10. Council vote sheet

```
ARCH-1 Decision: ____________________

[ ] 🟢 GREEN  — Inside-out approved. Tier-2 closeout precedes scout pass.
[ ] 🟡 YELLOW — Hybrid approved. Tech debt acknowledged.
[ ] 🔴 RED    — Outside-in stays. ARCH-5 closes.

Notes: ___________________________________________________________

___________________________________________________________________

Signed: Joe Ibrahim                              Date: ____________
```

---

**End of ARCH-1 brief.** Council session reads this alongside `02_ARCH-2_MONETA_EVOLUTION_BRIEF.md` and `04_ARCH-5_WORKFLOW_LIVRPS_BRIEF.md`, then decides all three in single sitting.
