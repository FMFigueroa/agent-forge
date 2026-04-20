import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_forge import __version__
from agent_forge.observability.factory import build_pipeline, build_tracer

app = typer.Typer(
    name="agent-forge",
    help="Multi-agent LinkedIn content factory powered by Claude.",
    no_args_is_help=True,
)
traces_app = typer.Typer(help="Inspect agent traces.", no_args_is_help=True)
app.add_typer(traces_app, name="traces")
console = Console()


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"[bold cyan]agent-forge[/] v{__version__}")


@app.command()
def generate(
    topic: str = typer.Argument(..., help="The topic for the LinkedIn post."),
    persona: str = typer.Option("senior-engineer", "--persona", "-p"),
) -> None:
    """Generate a LinkedIn post by orchestrating the agent team."""
    console.print(f"[yellow]→ Generating post about:[/] {topic}")
    console.print(f"[yellow]→ Persona:[/] {persona}")
    console.print("[dim]Running researcher → drafter → orchestrator...[/]")

    pipeline = build_pipeline()
    result = asyncio.run(pipeline.run(topic))

    console.print()
    console.print(Panel(result.final_post, title="Final post", border_style="green"))
    console.print(
        f"\n[dim]→ Run [cyan]{result.run_id}[/] traced. "
        f"Cost: [green]${result.total_cost:.4f}[/]. "
        f"Inspect: [bold]agent-forge traces show {result.run_id}[/][/]"
    )


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server. (stub)"""
    console.print(f"[yellow]→ Will start server on {host}:{port}[/]")
    console.print("[red]Not yet implemented — coming soon.[/]")


@traces_app.command("ls")
def traces_ls(limit: int = typer.Option(10, "--limit", "-n", min=1, max=500)) -> None:
    """List the most recent runs with span count and total cost."""
    tracer = build_tracer()
    runs = tracer.list_runs(limit=limit)

    if not runs:
        console.print("[dim]No runs recorded yet.[/]")
        return

    table = Table(title=f"Recent runs (last {len(runs)})")
    table.add_column("run_id", style="cyan", no_wrap=True)
    table.add_column("created_at", style="dim")
    table.add_column("status")
    table.add_column("spans", justify="right")
    table.add_column("cost (USD)", justify="right", style="green")

    for r in runs:
        status_color = {"running": "yellow", "completed": "green", "failed": "red"}.get(
            r.status, "white"
        )
        table.add_row(
            r.run_id,
            r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            f"[{status_color}]{r.status}[/]",
            str(r.span_count),
            f"${r.total_cost:.4f}",
        )

    console.print(table)


@traces_app.command("show")
def traces_show(run_id: str = typer.Argument(..., help="Run ID to inspect.")) -> None:
    """Show all spans for a given run."""
    tracer = build_tracer()
    spans = tracer.by_run(run_id)

    if not spans:
        console.print(f"[red]No spans found for run_id={run_id!r}[/]")
        raise typer.Exit(code=1)

    total_cost = sum(s.cost_usd for s in spans)
    total_in = sum(s.usage.input_tokens for s in spans)
    total_out = sum(s.usage.output_tokens for s in spans)
    total_cache_read = sum(s.usage.cache_read_input_tokens for s in spans)

    summary = (
        f"[bold cyan]{run_id}[/]\n"
        f"spans: {len(spans)}   "
        f"cost: [green]${total_cost:.4f}[/]   "
        f"tokens: {total_in} in / {total_out} out"
    )
    if total_cache_read:
        summary += f"   [dim]cache read: {total_cache_read}[/]"
    console.print(Panel(summary, title="Run", expand=False))

    table = Table(title="Spans")
    table.add_column("agent", style="cyan")
    table.add_column("model", style="dim")
    table.add_column("latency (ms)", justify="right")
    table.add_column("in", justify="right")
    table.add_column("out", justify="right")
    table.add_column("cache r/w", justify="right", style="dim")
    table.add_column("cost (USD)", justify="right", style="green")
    table.add_column("stop", style="dim")

    for s in spans:
        stop = s.error or s.stop_reason or "-"
        stop_style = "red" if s.error else "dim"
        table.add_row(
            s.agent_name,
            s.model,
            f"{s.latency_ms:.0f}",
            str(s.usage.input_tokens),
            str(s.usage.output_tokens),
            f"{s.usage.cache_read_input_tokens}/{s.usage.cache_creation_input_tokens}",
            f"${s.cost_usd:.4f}",
            f"[{stop_style}]{stop}[/]",
        )

    console.print(table)


if __name__ == "__main__":
    app()
