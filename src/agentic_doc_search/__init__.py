"""Vectorless agentic document search.

An LLM agent that navigates and reasons through document corpora
without embeddings or indexes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_doc_search.corpus import Corpus
from agentic_doc_search.models import SearchResult

if TYPE_CHECKING:
    from agentic_doc_search.agent import Agent

__all__ = ["Agent", "Corpus", "SearchResult"]
__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    if name == "Agent":
        from agentic_doc_search.agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
