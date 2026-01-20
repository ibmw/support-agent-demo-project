"""
Typer CLI for the Support Agent.

Commands:
    index               Index Help Center articles into ChromaDB
    chat                Interactive terminal chat with the agent
    serve               Launch Gradio UI (not yet implemented)
    evaluate            Run single-turn evaluation
    evaluate-conversations  Run multi-turn conversation evaluation
"""

from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from support_agent.agent import AgentResponse, get_support_crew
from support_agent.config import HELP_CSV, QUESTIONS_CSV, SCENARIOS_JSON
from support_agent.evaluation import (
    ConversationEvaluationRunner,
    EvaluationRunner,
    format_conversation_summary_report,
    format_summary_report,
)
from support_agent.indexing import (
    HelpCenterParser,
    KnowledgeBaseIndexer,
    SemanticChunker,
)
from support_agent.logging import get_logger, setup_logging

app = typer.Typer(
    name="support-agent",
    help="Customer Support AI Agent CLI",
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__, component="cli")


# =============================================================================
# Index Command
# =============================================================================


@app.command()
def index(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Clear existing index and rebuild"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Index Help Center articles into ChromaDB.

    Parses articles from the CSV, chunks them semantically, and indexes
    them with OpenAI embeddings.
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    rprint("\n[bold blue]📚 Knowledge Base Indexer[/bold blue]\n")

    try:
        # Initialize components
        parser = HelpCenterParser()
        chunker = SemanticChunker()
        indexer = KnowledgeBaseIndexer()

        # Clear index if force flag is set
        if force:
            rprint("[yellow]⚠️  Clearing existing index...[/yellow]")
            indexer.clear_collection()
            rprint("[green]✓ Index cleared[/green]\n")

        # Check current index status
        stats = indexer.get_collection_stats()
        if stats["count"] > 0 and not force:
            rprint(
                f"[yellow]ℹ️  Index already contains {stats['count']} chunks.[/yellow]"
            )
            rprint("[yellow]   Use --force to rebuild from scratch.[/yellow]\n")

            if not typer.confirm("Continue indexing anyway?"):
                raise typer.Abort()

        # Load and parse articles
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading articles from CSV...", total=None)
            articles = parser.load_articles()

        rprint(f"[green]✓ Loaded {len(articles)} articles from {HELP_CSV.name}[/green]")

        # Chunk articles
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Chunking articles...", total=None)
            chunks = chunker.chunk_all(articles)

        rprint(f"[green]✓ Created {len(chunks)} chunks[/green]")

        # Index chunks
        rprint("\n[bold]Indexing chunks (this may take a while)...[/bold]\n")
        indexer.index_chunks(chunks)

        # Final stats
        final_stats = indexer.get_collection_stats()
        rprint("\n[green]✓ Indexing complete![/green]")
        rprint(f"  Collection: {final_stats['name']}")
        rprint(f"  Total chunks: {final_stats['count']}")

    except typer.Abort:
        rprint("\n[yellow]Aborted.[/yellow]")
    except Exception as e:
        logger.error("Indexing failed", error=str(e), error_type=type(e).__name__)
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Chat Command
# =============================================================================


def _format_response(response: AgentResponse) -> Panel:
    """Format an agent response as a rich Panel."""
    action_colors = {
        "CLOSE": "green",
        "HANDOVER": "yellow",
        "WAIT": "blue",
    }
    color = action_colors.get(response.action, "white")

    content = f"{response.message}"

    if response.sources:
        content += f"\n\n[dim]Sources: {', '.join(response.sources)}[/dim]"

    if response.confidence is not None:
        content += f"\n[dim]Confidence: {response.confidence:.0%}[/dim]"

    return Panel(
        content,
        title=f"[{color}]{response.action}[/{color}]",
        border_style=color,
    )


@app.command()
def chat(
    query: Annotated[
        str | None,
        typer.Argument(help="Query to send (omit for interactive mode)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Chat with the support agent.

    Provide a query as an argument for single-shot mode, or omit
    for interactive conversation mode.

    Examples:
        uv run python -m cli.main chat "How do I reset my password?"
        uv run python -m cli.main chat  # Interactive mode
    """
    setup_logging(level="DEBUG" if verbose else "WARNING")

    rprint("\n[bold blue]🤖 Support Agent Chat[/bold blue]\n")

    try:
        crew = get_support_crew(verbose=verbose)

        if query:
            # Single-shot mode
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Processing query...", total=None)
                response = crew.process_query(query)

            rprint(_format_response(response))
        else:
            # Interactive mode
            rprint("[dim]Type 'quit' or 'exit' to end the conversation.[/dim]")
            rprint("[dim]Type 'new' to start a new conversation.[/dim]\n")

            session_id = None

            while True:
                try:
                    user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
                except (KeyboardInterrupt, EOFError):
                    rprint("\n[yellow]Goodbye![/yellow]")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit"):
                    rprint("[yellow]Goodbye![/yellow]")
                    break

                if user_input.lower() == "new":
                    session_id = None
                    rprint("[green]Started new conversation.[/green]\n")
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Thinking...", total=None)
                    response, session_id = crew.process_conversation(
                        user_input, session_id
                    )

                rprint()
                rprint(_format_response(response))
                rprint()

    except Exception as e:
        logger.error("Chat failed", error=str(e), error_type=type(e).__name__)
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Serve Command
# =============================================================================


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind the server to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to run the API server on"),
    ] = 8000,
    ui_port: Annotated[
        int,
        typer.Option("--ui-port", help="Port to run Gradio UI on"),
    ] = 7860,
    api_only: Annotated[
        bool,
        typer.Option("--api-only", help="Start FastAPI only (no Gradio UI)"),
    ] = False,
    share: Annotated[
        bool,
        typer.Option("--share", help="Create a public Gradio share link"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Start the FastAPI server and Gradio UI.

    Examples:
        uv run python -m cli.main serve                    # Start API + Gradio UI
        uv run python -m cli.main serve --api-only         # API only (no Gradio)
        uv run python -m cli.main serve --port 8080        # Custom API port
        uv run python -m cli.main serve --share            # Create public share link
    """
    import threading

    import uvicorn

    setup_logging(level="DEBUG" if verbose else "INFO")

    rprint("\n[bold blue]🚀 Support Agent Server[/bold blue]\n")

    if api_only:
        rprint(f"[dim]Starting FastAPI server on http://{host}:{port}[/dim]")
        rprint("[dim]API docs available at /docs[/dim]\n")

        uvicorn.run(
            "support_agent.api.main:app",
            host=host,
            port=port,
            log_level="debug" if verbose else "info",
        )
    else:
        from support_agent.ui.app import create_gradio_app, launch_app

        # Start FastAPI in a background thread
        def run_api():
            uvicorn.run(
                "support_agent.api.main:app",
                host=host,
                port=port,
                log_level="warning",  # Quieter for combined mode
            )

        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()

        rprint(f"[green]✓[/green] API server running on http://{host}:{port}")
        rprint(f"[dim]  API docs: http://{host}:{port}/docs[/dim]")

        # Start Gradio UI (blocking)
        rprint(f"[green]✓[/green] Gradio UI starting on http://{host}:{ui_port}")
        if share:
            rprint("[dim]  Creating public share link...[/dim]")
        rprint()

        gradio_app = create_gradio_app(api_base_url=f"http://{host}:{port}")
        # Gradio 6.0: css and theme must be passed to launch()
        launch_app(
            gradio_app,
            server_name=host,
            server_port=ui_port,
            share=share,
            show_error=True,
            quiet=not verbose,
        )


# =============================================================================
# Evaluate Command
# =============================================================================


@app.command()
def evaluate(
    sample: Annotated[
        int | None,
        typer.Option("--sample", "-n", help="Number of questions to sample"),
    ] = None,
    all_questions: Annotated[
        bool,
        typer.Option("--all", "-a", help="Evaluate all questions"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (CSV or JSON)"),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", "-t", help="Tag for this evaluation run"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Run single-turn evaluation on test questions.

    Examples:
        uv run python -m cli.main evaluate --sample 10
        uv run python -m cli.main evaluate --all --output results/eval.csv
        uv run python -m cli.main evaluate --sample 50 --tag "experiment-1"
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    rprint("\n[bold blue]📊 Single-Turn Evaluation[/bold blue]\n")

    if not sample and not all_questions:
        rprint("[yellow]Please specify --sample N or --all[/yellow]")
        raise typer.Exit(1)

    try:
        # Determine output format from extension
        output_format = "csv"
        if output and output.suffix.lower() == ".json":
            output_format = "json"

        # Create runner
        runner = EvaluationRunner(
            questions_path=QUESTIONS_CSV,
            sample_size=None if all_questions else sample,
            random_seed=seed,
            eval_tag=tag,
            output_format=output_format,
        )

        total = len(runner.questions)
        eval_count = total if all_questions else (sample or total)
        rprint(f"[dim]Questions: {eval_count} / {total}[/dim]")
        if tag:
            rprint(f"[dim]Tag: {tag}[/dim]")
        rprint()

        # Progress callback for live updates
        def progress_callback(current: int, total: int, result) -> None:
            status = "✓" if result.success else "✗"
            action = result.response.action if result.response else "ERROR"
            rprint(
                f"  [{current}/{total}] {status} {action} - {result.question[:50]}..."
            )

        # Run evaluation
        results, summary = runner.run(progress_callback=progress_callback)

        # Print summary
        rprint("\n" + "=" * 60)
        rprint(format_summary_report(summary))

        # Export if output specified
        if output:
            runner.export(results, summary, output)
            rprint(f"\n[green]✓ Results exported to {output}[/green]")

    except Exception as e:
        logger.error("Evaluation failed", error=str(e), error_type=type(e).__name__)
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Evaluate Conversations Command
# =============================================================================


@app.command("evaluate-conversations")
def evaluate_conversations(
    scenarios: Annotated[
        Path | None,
        typer.Option("--scenarios", "-s", help="Path to scenarios JSON file"),
    ] = None,
    ids: Annotated[
        str | None,
        typer.Option("--ids", help="Comma-separated scenario IDs to run"),
    ] = None,
    categories: Annotated[
        str | None,
        typer.Option("--categories", "-c", help="Comma-separated categories to filter"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (CSV or JSON)"),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", "-t", help="Tag for this evaluation run"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Run multi-turn conversation evaluation.

    Examples:
        uv run python -m cli.main evaluate-conversations
        uv run python -m cli.main evaluate-conversations --ids "password-reset-simple,refund-request-handover"
        uv run python -m cli.main evaluate-conversations --categories "account,billing"
        uv run python -m cli.main evaluate-conversations --output results/conv_eval.json
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    rprint("\n[bold blue]💬 Conversation Evaluation[/bold blue]\n")

    try:
        scenarios_path = scenarios or SCENARIOS_JSON

        # Parse filter options
        scenario_ids = [s.strip() for s in ids.split(",")] if ids else None
        category_list = (
            [c.strip() for c in categories.split(",")] if categories else None
        )

        # Determine output format
        output_format = "json"
        if output and output.suffix.lower() == ".csv":
            output_format = "csv"

        # Create runner
        runner = ConversationEvaluationRunner(
            scenarios_path=scenarios_path,
            scenario_ids=scenario_ids,
            categories=category_list,
            eval_tag=tag,
            output_format=output_format,
        )

        total = len(runner.scenarios)
        rprint(f"[dim]Scenarios: {total}[/dim]")
        if scenario_ids:
            rprint(f"[dim]Filtered by IDs: {scenario_ids}[/dim]")
        if category_list:
            rprint(f"[dim]Filtered by categories: {category_list}[/dim]")
        if tag:
            rprint(f"[dim]Tag: {tag}[/dim]")
        rprint()

        # Progress callback
        def progress_callback(current: int, total: int, result) -> None:
            status = "✓" if result.success else "✗"
            action = result.final_action or "N/A"
            rprint(
                f"  [{current}/{total}] {status} {action} - "
                f"{result.scenario_name} ({result.turn_count} turns)"
            )

        # Run evaluation
        results, summary = runner.run(progress_callback=progress_callback)

        # Print summary
        rprint("\n" + "=" * 60)
        rprint(format_conversation_summary_report(summary))

        # Export if output specified
        if output:
            runner.export(results, summary, output)
            rprint(f"\n[green]✓ Results exported to {output}[/green]")

    except Exception as e:
        logger.error(
            "Conversation evaluation failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Stats Command (bonus utility)
# =============================================================================


@app.command()
def stats() -> None:
    """
    Show knowledge base statistics.
    """
    rprint("\n[bold blue]📈 Knowledge Base Stats[/bold blue]\n")

    try:
        indexer = KnowledgeBaseIndexer()
        collection_stats = indexer.get_collection_stats()

        table = Table(title="ChromaDB Collection")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collection Name", collection_stats["name"])
        table.add_row("Total Chunks", str(collection_stats["count"]))

        console.print(table)

    except Exception as e:
        logger.error("Stats failed", error=str(e), error_type=type(e).__name__)
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
