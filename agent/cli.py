"""Command-line interface for the ComfyUI Agent."""

import atexit
import json
import logging
import signal
import threading

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ._conn_ctx import current_conn_session
from .config import AGENT_MODEL, COMFYUI_URL, COMFYUI_DATABASE, ANTHROPIC_API_KEY, LOG_DIR
from .logging_config import setup_logging
from .streaming import NullHandler
from .tools import comfy_inspect, comfy_discover, session_tools

log = logging.getLogger(__name__)

app = typer.Typer(
    help="ComfyUI Agent -- AI co-pilot for ComfyUI workflows",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# CLI stream handler
# ---------------------------------------------------------------------------


class CLIHandler(NullHandler):
    """Rich terminal output handler."""

    def __init__(self, rich_console: Console):
        self._console = rich_console
        self._streamed_any = False

    def on_text(self, text: str) -> None:
        self._streamed_any = True
        print(text, end="", flush=True)

    def on_tool_call(self, name: str, input: dict) -> None:
        if self._streamed_any:
            print()
            self._streamed_any = False
        inp_summary = json.dumps(input, default=str, sort_keys=True, allow_nan=False)  # Cycle 61
        if len(inp_summary) > 80:
            inp_summary = inp_summary[:77] + "..."
        self._console.print(f"  [dim]-> {name}({inp_summary})[/dim]")

    def on_stream_end(self) -> None:
        if self._streamed_any:
            print()
            self._streamed_any = False

    def on_input(self) -> str | None:
        try:
            return self._console.input("[bold cyan]> [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            self._console.print()
            return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    session: str = typer.Option(
        None, "--session", "-s",
        help="Named session for memory persistence",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show debug logging (API timing, tool execution, context management)",
    ),
):
    """Start an interactive agent session.

    Examples:
      agent run                       # ephemeral session, no memory
      agent run --session portrait    # named session, remembers across calls
      agent run -v                    # verbose: API timing + tool tracing
    """
    # Configure logging
    setup_logging(
        level=logging.DEBUG if verbose else logging.WARNING,
        log_file=LOG_DIR / "agent.log",
    )
    # Validate API key
    if not ANTHROPIC_API_KEY:
        console.print(
            "[red]ANTHROPIC_API_KEY not set.[/red]\n"
            "Setup steps:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Get a key from https://console.anthropic.com/\n"
            "  3. Add: ANTHROPIC_API_KEY=sk-ant-...\n"
        )
        raise typer.Exit(1)

    # Show header
    console.print(Panel.fit(
        f"[bold blue]ComfyUI Agent[/bold blue]\n"
        f"Model: {AGENT_MODEL}\n"
        f"ComfyUI: {COMFYUI_URL}\n"
        f"Database: {COMFYUI_DATABASE}",
        title="ComfyUI Agent v0.4",
    ))

    # Load session context for system prompt injection
    session_context = None
    if session:
        console.print(f"[dim]Session: {session}[/dim]")
        # Restore session state
        load_result = json.loads(session_tools.handle("load_session", {"name": session}))
        if "error" not in load_result:
            console.print(f"[dim]Restored session from {load_result.get('saved_at', '?')}[/dim]")
            if load_result.get("workflow_restored"):
                console.print(f"[dim]Workflow loaded: {load_result.get('workflow_path')}[/dim]")
            if load_result.get("notes_count", 0) > 0:
                console.print(f"[dim]{load_result['notes_count']} note(s) from previous session[/dim]")
            # Build session context for system prompt
            session_context = {
                "name": session,
                "notes": load_result.get("notes", []),
                "workflow": {
                    "loaded_path": load_result.get("workflow_path"),
                    "format": load_result.get("workflow_format"),
                    "history_depth": 0,
                },
            }

            # Detect last output image for metadata resume
            if session_context and load_result.get("workflow_restored"):
                try:
                    from .config import COMFYUI_OUTPUT_DIR
                    pngs = sorted(
                        COMFYUI_OUTPUT_DIR.glob("*.png"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if pngs:
                        session_context["last_output_path"] = str(pngs[0])
                except OSError as e:
                    log.debug("Failed to detect last output image: %s", e)

    console.print("[dim]Type your question or command. 'quit' to exit.[/dim]\n")

    # Import here to avoid loading anthropic at CLI startup
    from .main import create_client, run_interactive, request_shutdown

    client = create_client()

    # Graceful shutdown: save session on SIGTERM or atexit
    _shutdown_done = threading.Event()

    def _save_and_exit(signum=None, frame=None):
        if _shutdown_done.is_set():
            return
        _shutdown_done.set()
        request_shutdown()
        if session:
            try:
                # Cycle 16 hotfix: ensure the _conn_session ContextVar matches
                # the session we're saving. _save_and_exit can run from atexit
                # or a SIGTERM handler AFTER cli.run's try/finally has reset
                # the contextvar — without this self-set, save_session would
                # read get_session("default") (the empty default workflow
                # state) instead of the user's session, silently corrupting
                # the user's foo.json on every normal exit.
                from ._conn_ctx import _conn_session
                _conn_session.set(session)
                save_result = json.loads(
                    session_tools.handle("save_session", {"name": session})
                )
                if "saved" in save_result:
                    console.print(f"\n[dim]Session '{session}' saved.[/dim]")
            except Exception as e:
                log.warning(
                    "Failed to save session '%s' on exit: %s", session, e
                )

    # Cycle 15: thread the --session flag through the _conn_session ContextVar
    # so every tool call inside run_interactive sees the right session.  Cycles
    # 0+1+4+12 fixed sidebar / MCP / panel / stage modules; the CLI was the
    # last transport that wasn't setting the contextvar, so `agent run --session
    # foo` was functionally equivalent to `agent run` for tool isolation.
    from ._conn_ctx import _conn_session
    from .logging_config import set_correlation_id

    sid = session or "default"
    _conn_session_token = _conn_session.set(sid)
    set_correlation_id(sid)

    # Cycle 16 hotfix: register signal/atexit AFTER the contextvar is set,
    # so a SIGTERM that fires during startup runs _save_and_exit with the
    # contextvar already pointing at the right session.
    signal.signal(signal.SIGTERM, _save_and_exit)
    atexit.register(
        lambda: _save_and_exit() if session and not _shutdown_done.is_set() else None
    )

    handler = CLIHandler(console)
    try:
        run_interactive(client, session_context=session_context, handler=handler)
        # Cycle 16 hotfix: call _save_and_exit BEFORE the finally block
        # resets the contextvar. _save_and_exit's own self-set is the
        # belt; this is the suspenders — it ensures the normal-exit path
        # never sees a reset contextvar.
        _save_and_exit()
    finally:
        _conn_session.reset(_conn_session_token)

    console.print("\n[dim]Goodbye![/dim]")


@app.command()
def inspect():
    """Quick summary of the local ComfyUI installation.

    Reports installed models (counts by family), custom node packs, and
    a check for whether ComfyUI itself is reachable. No API key needed.

    Examples:
      agent inspect                   # full summary
    """
    console.print("[bold]ComfyUI Installation Summary[/bold]\n")

    # Models summary
    result = json.loads(comfy_inspect.handle("get_models_summary", {}))
    if "error" in result:
        console.print(f"[red]Models: {result['error']}[/red]")
    else:
        console.print(f"[bold]Models[/bold] ({result['directory']})")
        table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")
        for model_type, count in sorted(result["types"].items()):
            table.add_row(model_type, str(count))
        console.print(table)
        console.print()

    # Custom nodes
    result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))
    if "error" in result:
        console.print(f"[red]Custom nodes: {result['error']}[/red]")
    else:
        console.print(f"[bold]Custom Node Packs[/bold] ({result['count']} installed)")
        table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        table.add_column("Pack Name", style="bold")
        table.add_column("Nodes", justify="center")
        table.add_column("Readme", justify="center")
        for pack in result["packs"]:
            nodes_mark = "yes" if pack.get("registers_nodes") else ""
            readme_mark = "yes" if pack.get("has_readme") else ""
            table.add_row(pack["name"], nodes_mark, readme_mark)
        console.print(table)


@app.command()
def parse(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
):
    """Analyze a ComfyUI workflow file.

    Reports node count, format detection (API / UI+API / UI-only),
    sampler parameters, and any deprecation warnings. Read-only —
    never modifies the file.

    Examples:
      agent parse my_workflow.json
      agent parse ~/Downloads/sdxl_portrait.json
    """
    from pathlib import Path

    path = Path(workflow)
    if not path.exists():
        console.print(f"[red]File not found: {workflow}[/red]")
        raise typer.Exit(1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        raise typer.Exit(1)

    # Detect format and extract nodes
    if "nodes" in data and isinstance(data["nodes"], list):
        fmt = "UI format"
        # Check for embedded API format in extra.prompt
        api_data = data.get("extra", {}).get("prompt")
        if api_data and isinstance(api_data, dict):
            # Has embedded API format
            nodes = {
                k: v for k, v in api_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
        else:
            # UI-only -- convert nodes array to dict keyed by ID
            nodes = {}
            for node in data["nodes"]:
                nid = str(node.get("id", ""))
                node_type = node.get("type", "Unknown")
                widgets = node.get("widgets_values", [])
                nodes[nid] = {
                    "class_type": node_type,
                    "inputs": {},  # Can't fully reconstruct without object_info
                    "_widgets_values": widgets,
                }
    else:
        fmt = "API format"
        nodes = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v
        }

    console.print(f"[bold]Workflow:[/bold] {path.name}")
    console.print(f"[bold]Format:[/bold] {fmt}")
    console.print(f"[bold]Nodes:[/bold] {len(nodes)}")
    console.print()

    # List nodes by class type
    class_counts: dict[str, int] = {}
    for node in nodes.values():
        ct = node.get("class_type", "Unknown")
        class_counts[ct] = class_counts.get(ct, 0) + 1

    table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
    table.add_column("Node Type", style="cyan")
    table.add_column("Count", justify="right")
    for ct, count in sorted(class_counts.items()):
        table.add_row(ct, str(count))
    console.print("[bold]Node types:[/bold]")
    console.print(table)

    # Show editable fields (API format only)
    api_nodes = {
        nid: n for nid, n in nodes.items()
        if n.get("inputs") and not n.get("_widgets_values")
    }
    if api_nodes:
        fields_table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        fields_table.add_column("Node [ID]", style="bold")
        fields_table.add_column("Field")
        fields_table.add_column("Value", style="dim")
        for nid, node in sorted(api_nodes.items()):
            ct = node.get("class_type", "?")
            for fld, value in node.get("inputs", {}).items():
                if isinstance(value, list):
                    continue  # connection
                val_repr = repr(value)
                if len(val_repr) > 50:
                    val_repr = val_repr[:47] + "..."
                fields_table.add_row(f"{ct} [{nid}]", fld, val_repr)
        console.print("\n[bold]Editable fields:[/bold]")
        console.print(fields_table)
    elif fmt == "UI format":
        console.print(
            "\n[dim]UI-format workflow -- editable fields require ComfyUI "
            "running (use interactive mode to analyze with /object_info).[/dim]"
        )


@app.command()
def sessions():
    """List saved agent sessions.

    Sessions are created when you pass `--session NAME` to
    `agent run`. They persist memory, ratchet history, and learned
    parameter preferences across invocations.

    Examples:
      agent sessions                  # list all saved sessions
    """
    result = json.loads(session_tools.handle("list_sessions", {}))
    if result["count"] == 0:
        console.print("[dim]No saved sessions. Use --session NAME with 'run' to create one.[/dim]")
        return

    console.print(f"[bold]Saved Sessions[/bold] ({result['count']})\n")
    table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
    table.add_column("Name", style="bold")
    table.add_column("Saved At")
    table.add_column("Workflow", justify="center")
    table.add_column("Notes", justify="right")
    for s in result["sessions"]:
        table.add_row(
            s["name"],
            s.get("saved_at", "?"),
            "yes" if s.get("has_workflow") else "",
            str(s.get("notes_count", 0)) if s.get("notes_count") else "",
        )
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for models or nodes"),
    nodes: bool = typer.Option(False, "--nodes", "-n", help="Search custom node packs"),
    models: bool = typer.Option(False, "--models", "-m", help="Search model registry"),
    node_type: bool = typer.Option(
        False, "--node-type", "-t",
        help="Search by specific node class_type",
    ),
    huggingface: bool = typer.Option(
        False, "--hf", help="Search HuggingFace instead of local registry",
    ),
    model_type: str = typer.Option(
        None, "--type",
        help="Filter models by type (checkpoint, lora, vae, controlnet)",
    ),
):
    """Search for custom nodes or models.

    Searches the local ComfyUI Manager registry by default. Use --hf to
    search HuggingFace, --type to filter by model kind. No API key needed
    for local registry.

    Examples:
      agent search "flux dev"                  # local registry, all categories
      agent search "ipadapter" --nodes         # only custom node packs
      agent search "sdxl" --models             # only models
      agent search "lora" --models --type lora # only LoRA models
      agent search "flux" --hf                 # HuggingFace search
    """
    # Build discover params
    params: dict = {"query": query}

    if node_type or nodes:
        params["category"] = "nodes"
    elif models:
        params["category"] = "models"

    if huggingface:
        params["sources"] = ["huggingface"]

    if model_type:
        params["model_type"] = model_type

    result = json.loads(comfy_discover.handle("discover", params))
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    if result.get("results"):
        console.print(
            f"[bold]{result['total']} results[/bold] "
            f"(sources: {', '.join(result.get('sources_searched', []))})\n"
        )
        table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        table.add_column("Name", style="bold")
        table.add_column("Type")
        table.add_column("Source")
        table.add_column("Status")
        for r in result["results"]:
            name = r.get("name", "?")
            rtype = r.get("type", "")
            source = r.get("source", "")
            extra = [
                p for p in [r.get("model_type", ""), r.get("base_model", "")] if p
            ]
            if extra:
                rtype = f"{rtype} ({', '.join(extra)})" if rtype else ", ".join(extra)
            status = "[green]installed[/green]" if r.get("installed") else ""
            table.add_row(name, rtype, source, status)
        console.print(table)

        # Show descriptions below the table for entries that have them
        descs = [
            (r.get("name", "?"), r.get("description", ""))
            for r in result["results"] if r.get("description")
        ]
        if descs:
            console.print()
            for name, desc in descs:
                console.print(f"  [bold]{name}[/bold]: {desc[:120]}")
    else:
        console.print(f"[yellow]No results for '{query}'[/yellow]")


@app.command()
def orchestrate(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
    session: str = typer.Option(
        None, "--session", "-s",
        help="Named session for state persistence",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show debug logging",
    ),
):
    """Autonomous pipeline: load > validate > execute > verify a workflow.

    Loads a workflow, validates it against ComfyUI, executes it, and
    verifies the output. Results are saved to the session if provided.

    Examples:
      agent orchestrate workflow.json
      agent orchestrate workflow.json --session daily_render
      agent orchestrate workflow.json --auto-repair       # install missing nodes
      agent orchestrate workflow.json -v                  # verbose tracing
    """
    from pathlib import Path

    setup_logging(
        level=logging.DEBUG if verbose else logging.WARNING,
        log_file=LOG_DIR / "orchestrate.log",
    )

    path = Path(workflow)
    if not path.exists():
        console.print(f"[red]File not found: {workflow}[/red]")
        raise typer.Exit(1)

    from .tools import comfy_execute, workflow_parse

    # Step 1: Load
    console.print("[bold]Step 1/4:[/bold] Loading workflow...")
    load_result = json.loads(
        workflow_parse.handle("load_workflow", {"path": str(path)})
    )
    if "error" in load_result:
        console.print(f"[red]Load failed: {load_result['error']}[/red]")
        raise typer.Exit(1)
    console.print(
        f"  Loaded {load_result.get('node_count', '?')} nodes "
        f"({load_result.get('format', '?')} format)"
    )

    # Step 2: Validate
    console.print("[bold]Step 2/4:[/bold] Validating...")
    val_result = json.loads(
        comfy_execute.handle("validate_before_execute", {})
    )
    if "error" in val_result:
        console.print(f"[red]Validation failed: {val_result['error']}[/red]")
        raise typer.Exit(1)
    if not val_result.get("valid", False):
        issues = val_result.get("issues", [])
        console.print(f"[yellow]Validation issues ({len(issues)}):[/yellow]")
        for issue in issues[:5]:
            console.print(f"  - {issue}")
        raise typer.Exit(1)
    console.print("  [green]Valid[/green]")

    # Step 3: Execute
    console.print("[bold]Step 3/4:[/bold] Executing...")
    exec_result = json.loads(
        comfy_execute.handle("execute_workflow", {})
    )
    if "error" in exec_result:
        console.print(f"[red]Execution failed: {exec_result['error']}[/red]")
        raise typer.Exit(1)
    prompt_id = exec_result.get("prompt_id", "?")
    console.print(f"  Queued as {prompt_id}")

    # Step 4: Verify
    console.print("[bold]Step 4/4:[/bold] Verifying output...")
    from .tools import verify_execution
    verify_result = json.loads(
        verify_execution.handle("verify_execution", {"prompt_id": prompt_id})
    )
    if "error" in verify_result:
        console.print(f"[yellow]Verify: {verify_result['error']}[/yellow]")
    else:
        status = verify_result.get("status", "unknown")
        console.print(f"  Status: {status}")
        outputs = verify_result.get("outputs", [])
        if outputs:
            for out in outputs[:3]:
                console.print(f"  Output: {out}")

    # Step 5: USD Scene Composition (if usd-core available)
    scene_composed = False
    try:
        from .stage import HAS_USD
        if HAS_USD:
            from .session_context import get_session_context
            ctx = get_session_context(session or current_conn_session())
            stage = ctx.ensure_stage()
            if stage is not None:
                console.print("[bold]Step 5/6:[/bold] Composing USD scene...")
                from .stage.compositor_tools import handle as comp_handle
                comp_result = json.loads(
                    comp_handle("compose_scene", {})
                )
                if "error" not in comp_result:
                    console.print("  [green]Scene composed[/green]")
                    scene_composed = True

                    # Validate scene
                    val = json.loads(comp_handle("validate_scene", {}))
                    if "overall" in val:
                        console.print(
                            f"  Scene quality: {val['overall']:.2f} "
                            f"(depth={val.get('depth_consistency', 0):.2f}, "
                            f"camera={val.get('camera_fidelity', 0):.2f})"
                        )
                else:
                    console.print(f"  [dim]Scene composition skipped: {comp_result.get('error', '?')}[/dim]")
    except ImportError as e:
        log.debug("USD scene composition unavailable: %s", e)

    # Step 6: Record experience (if FORESIGHT available)
    if scene_composed:
        try:
            ratchet = ctx.ensure_ratchet()
            if ratchet and ratchet.has_foresight:
                console.print("[bold]Step 6/6:[/bold] Recording experience...")
                ratchet.keep(
                    prompt_id,
                    {"aesthetic": 0.5},  # Placeholder scores
                    change_context={"action": "orchestrate_pipeline"},
                )
                console.print("  [green]Experience recorded[/green]")
        except Exception as e:
            log.debug("Failed to record FORESIGHT experience: %s", e)
    elif not scene_composed:
        console.print("[dim]Steps 5-6 skipped (usd-core not available)[/dim]")

    # Save session if requested
    if session:
        save_result = json.loads(
            session_tools.handle("save_session", {"name": session})
        )
        if "saved" in save_result:
            console.print(f"\n[dim]Session '{session}' saved.[/dim]")

    console.print("\n[bold green]Pipeline complete.[/bold green]")


@app.command()
def autoresearch(
    query: str = typer.Argument(
        None, help="What to search for (model, node, technique). "
        "Omit to run a FORESIGHT autoresearch pipeline with --program.",
    ),
    category: str = typer.Option(
        "all", "--category", "-c",
        help="Search category: nodes, models, or all",
    ),
    provision: bool = typer.Option(
        False, "--provision", "-p",
        help="Auto-provision (register in stage) uninstalled models found",
    ),
    program: str = typer.Option(
        None, "--program",
        help="Path to a program.md file for FORESIGHT autoresearch pipeline",
    ),
    budget_hours: float = typer.Option(
        1.0, "--budget-hours",
        help="Maximum runtime in hours for FORESIGHT pipeline",
    ),
    experiment_seconds: float = typer.Option(
        30.0, "--experiment-seconds",
        help="Expected seconds per experiment",
    ),
    max_experiments: int = typer.Option(
        100, "--max-experiments",
        help="Maximum number of experiments to run",
    ),
    report: bool = typer.Option(
        True, "--report/--no-report",
        help="Generate morning report at end",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume from saved session experience",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show debug logging",
    ),
):
    """Discover models/nodes or run FORESIGHT autoresearch pipeline.

    Without --program: searches all sources (registry, CivitAI, HuggingFace),
    displays results, and when --provision is set, registers uninstalled models
    in the CognitiveWorkflowStage model registry for later download.

    With --program: runs the FORESIGHT autoresearch pipeline — loads the
    program spec, initializes the ratchet with CWM+experience+arbiter,
    runs experiments, generates counterfactuals, and produces a morning report.

    Examples:
      agent autoresearch "flux dev"                       # discovery search
      agent autoresearch "sdxl realistic" --provision     # discover + queue downloads
      agent autoresearch --program research.md            # FORESIGHT pipeline run
      agent autoresearch --program research.md --hours 8  # bounded experiment budget
    """
    setup_logging(
        level=logging.DEBUG if verbose else logging.WARNING,
        log_file=LOG_DIR / "autoresearch.log",
    )

    # FORESIGHT autoresearch pipeline mode
    if program is not None:
        from pathlib import Path
        from .stage.autoresearch_runner import AutoresearchRunner, RunnerConfig

        if not Path(program).exists():
            console.print(f"[red]Program file not found: {program}[/red]")
            raise typer.Exit(1)

        config = RunnerConfig(
            budget_hours=budget_hours,
            experiment_seconds=experiment_seconds,
            max_experiments=max_experiments,
            program_path=program,
            session_name="autoresearch",
            resume=resume,
        )

        # Optionally wire CWS for experience persistence
        cws = None
        try:
            from .session_context import get_session_context
            ctx = get_session_context(current_conn_session())
            cws = ctx.ensure_stage()
        except Exception as e:
            log.debug("Could not initialize CWS for autoresearch: %s", e)

        console.print("[bold]FORESIGHT Autoresearch[/bold]")
        console.print(f"  Program: {program}")
        console.print(f"  Budget: {budget_hours}h / {max_experiments} experiments")
        console.print()

        runner = AutoresearchRunner(config, cws=cws)
        result = runner.run()

        console.print("[bold]Results:[/bold]")
        console.print(f"  Experiments: {len(result.experiments)}")
        console.print(
            f"  Kept: {sum(1 for e in result.experiments if e.kept)} / "
            f"{len(result.experiments)}"
        )
        console.print(f"  Stopped: {result.stopped_reason}")

        if report and result.report:
            console.print(f"\n{result.report}")

        return

    # Discovery mode (original behavior) — query is required
    if query is None:
        console.print("[red]Provide a search query or use --program for FORESIGHT mode.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Researching:[/bold] {query} (category={category})\n")

    result = json.loads(
        comfy_discover.handle("discover", {"query": query, "category": category})
    )
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    results = result.get("results", [])
    if not results:
        console.print(f"[yellow]No results for '{query}'[/yellow]")
        raise typer.Exit(0)

    # Display results
    table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Source")
    table.add_column("Status")
    for i, r in enumerate(results, 1):
        status = "[green]installed[/green]" if r.get("installed") else "[dim]not installed[/dim]"
        table.add_row(str(i), r.get("name", "?"), r.get("type", ""), r.get("source", ""), status)
    console.print(table)
    console.print(f"\n[dim]{len(results)} results from {', '.join(result.get('sources_searched', []))}[/dim]")

    # Auto-provision uninstalled models into the stage registry
    if provision:
        uninstalled = [r for r in results if not r.get("installed") and r.get("type") == "model"]
        if not uninstalled:
            console.print("\n[dim]All models already installed — nothing to provision.[/dim]")
            return

        console.print(f"\n[bold]Provisioning {len(uninstalled)} model(s) into stage registry...[/bold]")
        try:
            from .stage import HAS_USD
            if not HAS_USD:
                console.print("[yellow]usd-core not installed — cannot register in stage.[/yellow]")
                return

            from .session_context import get_session_context
            from .stage import register_model

            ctx = get_session_context(current_conn_session())
            stage = ctx.ensure_stage()
            if stage is None:
                console.print("[yellow]Could not initialize stage.[/yellow]")
                return

            for r in uninstalled:
                name = r.get("name", "unknown")
                model_type = r.get("model_type", "checkpoints")
                url = r.get("url", "")
                prim = register_model(
                    stage, model_type, name,
                    source_url=url,
                )
                console.print(f"  Registered: {prim}")

        except Exception as e:
            console.print(f"[red]Provision error: {e}[/red]")

    console.print()


@app.command()
def autonomous(
    hours: float = typer.Option(
        24.0, "--hours", "-h",
        help="Maximum runtime in hours.",
    ),
    max_experiments: int = typer.Option(
        1000, "--max-experiments",
        help="Maximum number of experiments before halting.",
    ),
    program: str = typer.Option(
        None, "--program",
        help="Path to a program.md file for the autoresearch driver.",
    ),
    workflow: str = typer.Option(
        None, "--workflow", "-w",
        help=(
            "Path to a workflow JSON. Required for --execute-mode real; "
            "ignored otherwise."
        ),
    ),
    execute_mode: str = typer.Option(
        "dry-run", "--execute-mode",
        help=(
            "Execution dispatch: 'dry-run' (real proposals + synthetic "
            "scores; no ComfyUI; the default — useful for smoke-testing "
            "the harness loop offline), 'real' (requires --workflow; "
            "mutates and executes against ComfyUI; failures degrade to "
            "zero scores so the ratchet rejects without halting), 'mock' "
            "(no callbacks; AutoresearchRunner falls back to neutral "
            "scores every iteration — for harness-internals testing only)."
        ),
    ),
    checkpoint_path: str = typer.Option(
        None, "--checkpoint",
        help="Override STAGE_DEFAULT_PATH for this run's checkpoint target.",
    ),
    checkpoint_every_seconds: float = typer.Option(
        300.0, "--checkpoint-every-seconds",
        help="Time-based checkpoint interval (also flushes every N iterations).",
    ),
    session_name: str = typer.Option(
        "cozy_autonomous", "--session",
        help="Session name for ratchet/experience persistence.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume from a previously checkpointed session.",
    ),
):
    """Run the long-running self-healing harness (Cozy Constitution).

    Wraps AutoresearchRunner with the self_healing_ladder classifier,
    checkpoints the stage on every iteration boundary, and only halts on
    TERMINAL classifications. See .claude/COZY_CONSTITUTION.md for doctrine.
    """
    from rich.console import Console
    console = Console()

    if execute_mode not in ("mock", "dry-run", "real"):
        console.print(
            f"[red]Invalid --execute-mode: {execute_mode!r}. "
            f"Choose one of: mock, dry-run, real.[/red]"
        )
        raise typer.Exit(code=2)
    if execute_mode == "real" and not workflow:
        console.print(
            "[red]--execute-mode real requires --workflow PATH.[/red]"
        )
        raise typer.Exit(code=2)

    # T6 from the 5x review: path-validate user-supplied paths before they
    # flow into the harness. validate_path() blocks traversal outside the
    # configured sandbox dirs (COMFYUI_DATABASE etc., plus tempdir for
    # pytest). Returns an error string if rejected, None if accepted.
    from .tools._util import validate_path
    if workflow:
        err = validate_path(workflow)
        if err:
            console.print(f"[red]--workflow path rejected: {err}[/red]")
            raise typer.Exit(code=2)
    if checkpoint_path:
        err = validate_path(checkpoint_path)
        if err:
            console.print(f"[red]--checkpoint path rejected: {err}[/red]")
            raise typer.Exit(code=2)

    from .harness import (
        CozyLoop, CozyLoopConfig,
        make_execute_fn, make_propose_fn,
    )
    from .session_context import get_session_context

    ctx = get_session_context(session_name)
    cws = ctx.ensure_stage()
    if cws is None:
        console.print(
            "[red]Cannot start autonomous harness: usd-core is not installed."
            "[/red]"
        )
        raise typer.Exit(code=1)

    config = CozyLoopConfig(
        budget_hours=hours,
        max_experiments=max_experiments,
        program_path=program,
        checkpoint_path=checkpoint_path,
        checkpoint_every_seconds=checkpoint_every_seconds,
        session_name=session_name,
        resume=resume,
    )

    # Build callbacks. With --execute-mode dry-run (the default after T3
    # of the 5x review), make_propose_fn cycles real proposals and
    # make_execute_fn returns synthetic axis scores. With --execute-mode
    # mock, both callbacks stay None and AutoresearchRunner falls back to
    # its built-in cyclic proposer + a neutral-score executor (returns
    # {aesthetic: 0.5, lighting: 0.5} every iteration — useful for
    # harness-internals testing only; produces zero ratchet signal).
    propose_fn = None
    execute_fn = None
    if execute_mode == "mock":
        console.print(
            "[yellow]WARNING: --execute-mode=mock uses AutoresearchRunner "
            "defaults — neutral scores every iteration, no ratchet signal. "
            "This mode is for harness-internals testing only. Use "
            "--execute-mode=dry-run (the default) for offline smoke tests "
            "with real proposal cycles, or --execute-mode=real --workflow "
            "PATH for live ComfyUI runs.[/yellow]"
        )
    elif execute_mode in ("dry-run", "real"):
        try:
            propose_fn = make_propose_fn()
            execute_fn = make_execute_fn(execute_mode, workflow)
        except (ValueError, OSError) as exc:
            console.print(f"[red]Failed to build execute callable: {exc}[/red]")
            raise typer.Exit(code=2) from exc

    console.print(
        f"[bold]Cozy autonomous harness[/bold] starting "
        f"(budget={hours}h, max_experiments={max_experiments}, "
        f"mode={execute_mode}, "
        f"checkpoint={config.checkpoint_path or 'in-memory only'})"
    )
    loop = CozyLoop(
        config,
        cws=cws,
        propose_fn=propose_fn,
        execute_fn=execute_fn,
    )
    result = loop.run()

    console.print(
        f"\n[bold]Halt[/bold]: {result.halt_reason}\n"
        f"  iterations:    "
        f"{len(result.run_result.experiments) if result.run_result else 0}\n"
        f"  total_seconds: {result.total_seconds:.1f}\n"
        f"  blocker_path:  {result.blocker_path or '(none)'}"
    )


@app.command()
def mcp():
    """Primary integration -- exposes all tools via MCP for Claude Code.

    Starts the MCP server using stdio transport. Configure in your
    Claude Code settings (.claude/settings.json) to use these tools
    directly from Claude Code conversations.

    Examples:
      agent mcp                       # start MCP server (stdio transport)

    Typical Claude Code config to wire this up (.claude/settings.json):

      {
        "mcpServers": {
          "comfyui-agent": {
            "command": "agent",
            "args": ["mcp"],
            "cwd": "/path/to/Comfy-Cozy"
          }
        }
      }
    """
    from .mcp_server import main as mcp_main
    mcp_main()


if __name__ == "__main__":
    app()
