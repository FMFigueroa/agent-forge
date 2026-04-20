import typer
from rich.console import Console

from agent_forge import __version__

app = typer.Typer(
    name="agent-forge",
    help="Multi-agent LinkedIn content factory powered by Claude.",
    no_args_is_help=True,
)
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
    """Generate a LinkedIn post by orchestrating the agent team. (stub)"""
    console.print(f"[yellow]→ Generating post about:[/] {topic}")
    console.print(f"[yellow]→ Persona:[/] {persona}")
    console.print("[red]Not yet implemented — coming soon.[/]")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server. (stub)"""
    console.print(f"[yellow]→ Will start server on {host}:{port}[/]")
    console.print("[red]Not yet implemented — coming soon.[/]")


if __name__ == "__main__":
    app()
