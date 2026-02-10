"""Command-line interface for the ComfyUI Agent."""

import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import AGENT_MODEL, COMFYUI_URL, COMFYUI_DATABASE, ANTHROPIC_API_KEY
from .tools import comfy_inspect, comfy_discover, session_tools

app = typer.Typer(
    help="ComfyUI Agent — AI co-pilot for ComfyUI workflows",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    session: str = typer.Option(
        None, "--session", "-s",
        help="Named session for memory persistence",
    ),
):
    """Start an interactive agent session."""
    # Validate API key
    if not ANTHROPIC_API_KEY:
        console.print(
            "[red]ANTHROPIC_API_KEY not set. "
            "Copy .env.example to .env and add your key.[/red]"
        )
        raise typer.Exit(1)

    # Show header
    console.print(Panel.fit(
        f"[bold blue]ComfyUI Agent[/bold blue]\n"
        f"Model: {AGENT_MODEL}\n"
        f"ComfyUI: {COMFYUI_URL}\n"
        f"Database: {COMFYUI_DATABASE}",
        title="ComfyUI Agent v0.1",
    ))

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

    console.print("[dim]Type your question or command. 'quit' to exit.[/dim]\n")

    # Import here to avoid loading anthropic at CLI startup
    from .main import create_client, run_interactive

    client = create_client()

    def on_text(text: str):
        console.print(text)

    def on_tool_call(name: str, inp: dict):
        # Show tool call in dim
        inp_summary = json.dumps(inp, default=str)
        if len(inp_summary) > 80:
            inp_summary = inp_summary[:77] + "..."
        console.print(f"  [dim]→ {name}({inp_summary})[/dim]")

    def on_thinking(text: str):
        # Optionally show thinking indicator
        if text:
            preview = text[:60].replace("\n", " ")
            console.print(f"  [dim italic]thinking: {preview}...[/dim italic]")

    def on_user_input() -> str | None:
        try:
            return console.input("[bold cyan]> [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print()
            return None

    run_interactive(
        client,
        on_text=on_text,
        on_tool_call=on_tool_call,
        on_thinking=on_thinking,
        on_user_input=on_user_input,
    )

    # Auto-save session on exit
    if session:
        save_result = json.loads(session_tools.handle("save_session", {"name": session}))
        if "saved" in save_result:
            console.print(f"\n[dim]Session '{session}' saved.[/dim]")

    console.print("\n[dim]Goodbye![/dim]")


@app.command()
def inspect():
    """Quick summary of the local ComfyUI installation."""
    console.print("[bold]ComfyUI Installation Summary[/bold]\n")

    # Models summary
    result = json.loads(comfy_inspect.handle("get_models_summary", {}))
    if "error" in result:
        console.print(f"[red]Models: {result['error']}[/red]")
    else:
        console.print(f"[bold]Models[/bold] ({result['directory']})")
        for model_type, count in sorted(result["types"].items()):
            console.print(f"  {model_type}: {count} file(s)")
        console.print()

    # Custom nodes
    result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))
    if "error" in result:
        console.print(f"[red]Custom nodes: {result['error']}[/red]")
    else:
        console.print(f"[bold]Custom Node Packs[/bold] ({result['count']} installed)")
        for pack in result["packs"]:
            name = pack["name"]
            markers = []
            if pack.get("registers_nodes"):
                markers.append("nodes")
            if pack.get("has_readme"):
                markers.append("readme")
            suffix = f" [{', '.join(markers)}]" if markers else ""
            console.print(f"  {name}{suffix}")


@app.command()
def parse(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
):
    """Analyze a ComfyUI workflow file."""
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
            # UI-only — convert nodes array to dict keyed by ID
            nodes = {}
            for node in data["nodes"]:
                nid = str(node.get("id", ""))
                node_type = node.get("type", "Unknown")
                # Extract widget values as inputs
                inputs = {}
                for inp in node.get("inputs", []):
                    if isinstance(inp, dict) and "name" in inp:
                        pass  # connection inputs handled separately
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

    console.print("[bold]Node types:[/bold]")
    for ct, count in sorted(class_counts.items()):
        suffix = f" (×{count})" if count > 1 else ""
        console.print(f"  {ct}{suffix}")

    # Show editable fields (API format only — UI format lacks field names)
    api_nodes = {
        nid: n for nid, n in nodes.items()
        if n.get("inputs") and not n.get("_widgets_values")
    }
    if api_nodes:
        console.print("\n[bold]Editable fields:[/bold]")
        for nid, node in sorted(api_nodes.items()):
            ct = node.get("class_type", "?")
            for field, value in node.get("inputs", {}).items():
                if isinstance(value, list):
                    continue  # connection
                val_repr = repr(value)
                if len(val_repr) > 50:
                    val_repr = val_repr[:47] + "..."
                console.print(f"  [{nid}] {ct}.{field} = {val_repr}")
    elif fmt == "UI format":
        console.print(
            "\n[dim]UI-format workflow — editable fields require ComfyUI "
            "running (use interactive mode to analyze with /object_info).[/dim]"
        )


@app.command()
def sessions():
    """List saved agent sessions."""
    result = json.loads(session_tools.handle("list_sessions", {}))
    if result["count"] == 0:
        console.print("[dim]No saved sessions. Use --session NAME with 'run' to create one.[/dim]")
        return

    console.print(f"[bold]Saved Sessions[/bold] ({result['count']})\n")
    for s in result["sessions"]:
        name = s["name"]
        saved_at = s.get("saved_at", "?")
        notes = s.get("notes_count", 0)
        wf = " [workflow]" if s.get("has_workflow") else ""
        notes_str = f" [{notes} note(s)]" if notes else ""
        console.print(f"  [bold]{name}[/bold]  {saved_at}{wf}{notes_str}")


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
    """Search for custom nodes or models."""
    if node_type or nodes:
        by = "node_type" if node_type else "name"
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": query, "by": by,
        }))
        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            return

        if result.get("match") == "exact":
            pack = result["pack"]
            status = "[green]installed[/green]" if pack["installed"] else "[yellow]not installed[/yellow]"
            console.print(f"[bold]{result['node_type']}[/bold] -> {pack['title']} ({status})")
            console.print(f"  {pack['url']}")
            console.print(f"  {pack['node_count']} node types in this pack")
        elif result.get("results"):
            console.print(f"[bold]{result.get('total_matches', len(result['results']))} matches[/bold]\n")
            for r in result["results"]:
                title = r.get("pack_title") or r.get("title", "?")
                url = r.get("pack_url") or r.get("url", "")
                installed = r.get("installed", False)
                status = "[green]installed[/green]" if installed else ""
                desc = r.get("description", "")
                console.print(f"  [bold]{title}[/bold] {status}")
                if desc:
                    console.print(f"    {desc[:100]}")
                if url:
                    console.print(f"    [dim]{url}[/dim]")
        else:
            console.print(f"[yellow]No results for '{query}'[/yellow]")
    else:
        # Default: search models
        source = "huggingface" if huggingface else "registry"
        params = {"query": query, "source": source}
        if model_type:
            params["model_type"] = model_type

        result = json.loads(comfy_discover.handle("search_models", params))
        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            return

        if result.get("results"):
            console.print(
                f"[bold]{result.get('total_matches', len(result['results']))} matches[/bold] "
                f"(source: {result['source']})\n"
            )
            for r in result["results"]:
                name = r.get("name", "?")
                mtype = r.get("type", "")
                base = r.get("base", "")
                size = r.get("size", "")
                installed = r.get("installed", False)
                status = " [green]installed[/green]" if installed else ""

                console.print(f"  [bold]{name}[/bold]{status}")
                parts = [p for p in [mtype, base, size] if p]
                if parts:
                    console.print(f"    {' | '.join(parts)}")
                url = r.get("url", "")
                if url:
                    console.print(f"    [dim]{url}[/dim]")
        else:
            console.print(f"[yellow]No results for '{query}'[/yellow]")


if __name__ == "__main__":
    app()
