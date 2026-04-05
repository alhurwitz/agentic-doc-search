"""Command-line interface for agentic-doc-search."""

from __future__ import annotations

from typing import Any, Callable

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agentic_doc_search.agent import Agent
from agentic_doc_search.corpus import Corpus
from agentic_doc_search.llm import LLMConfig
from agentic_doc_search.models import ToolCall

console = Console()


def _make_tool_callback(verbose: bool) -> Callable[[ToolCall], None]:
    """Create a callback that prints tool calls."""

    def callback(tc: ToolCall) -> None:
        if verbose:
            console.print(f"  [dim]→ {tc.tool_name}({_summarize_args(tc.arguments)})[/dim]")
        else:
            console.print(f"  [dim]→ {tc.tool_name}[/dim]")

    return callback


def _summarize_args(args: dict[str, Any]) -> str:
    """Produce a short summary of tool arguments."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 40:
            val = val[:37] + "..."
        parts.append(f"{k}={val!r}")
    return ", ".join(parts)


@click.group()
@click.version_option(package_name="agentic-doc-search")
def main() -> None:
    """Vectorless agentic document search."""


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.argument("query")
@click.option(
    "-m",
    "--model",
    default="claude-sonnet-4-20250514",
    help="LLM model to use (anything litellm supports).",
    show_default=True,
)
@click.option(
    "--max-iterations",
    default=20,
    help="Maximum agent iterations.",
    show_default=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed tool call arguments.")
@click.option("--api-key", envvar="LLM_API_KEY", help="API key (or set LLM_API_KEY env var).")
@click.option("--api-base", help="API base URL (for local models).")
def search(
    path: str,
    query: str,
    model: str,
    max_iterations: int,
    verbose: bool,
    api_key: str | None,
    api_base: str | None,
) -> None:
    """Search a document corpus with an AI agent.

    PATH is the directory containing your documents.
    QUERY is your question about the documents.

    Example:
        ads search ./my-docs "What is the refund policy?"
    """
    corpus = Corpus(path=path)
    files = corpus.discover_files()
    console.print(f"[bold]Corpus:[/bold] {path} ({len(files)} files)")
    console.print(f"[bold]Query:[/bold] {query}")
    console.print()

    llm_config = LLMConfig(
        model=model,
        api_key=api_key,
        api_base=api_base,
    )

    agent = Agent(
        llm_config=llm_config,
        max_iterations=max_iterations,
        on_tool_call=_make_tool_callback(verbose),
    )

    console.print("[bold]Searching...[/bold]")
    result = agent.search(corpus, query)
    console.print()

    # Display the answer
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))

    # Display token usage
    if result.token_usage:
        console.print(
            f"\n[dim]Tokens: {result.token_usage.prompt_tokens:,} prompt "
            f"+ {result.token_usage.completion_tokens:,} completion "
            f"= {result.token_usage.total_tokens:,} total "
            f"({len(result.reasoning)} tool calls)[/dim]"
        )


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--depth", default=3, help="Max tree depth.", show_default=True)
def tree(path: str, depth: int) -> None:
    """Show the directory structure of a document corpus."""
    corpus = Corpus(path=path)
    console.print(corpus.tree(max_depth=depth))


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def files(path: str) -> None:
    """List all document files in a corpus."""
    corpus = Corpus(path=path)
    for f in corpus.discover_files():
        console.print(f)


if __name__ == "__main__":
    main()
