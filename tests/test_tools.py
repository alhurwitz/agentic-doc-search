"""Tests for tool definitions and execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_doc_search.corpus import Corpus
from agentic_doc_search.tools import ALL_TOOLS, execute_tool


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    (tmp_path / "doc.md").write_text("# Title\n\nSome content about refunds.\nLine 4.\n")
    (tmp_path / "faq.txt").write_text("Q: How do I get a refund?\nA: Contact support.\n")
    return Corpus(path=tmp_path)


class TestToolSchemas:
    def test_all_tools_have_openai_schema(self) -> None:
        for tool in ALL_TOOLS:
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_tool_names_are_unique(self) -> None:
        names = [t.name for t in ALL_TOOLS]
        assert len(names) == len(set(names))

    def test_tool_serialization(self) -> None:
        """Tools should be serializable via Pydantic."""
        for tool in ALL_TOOLS:
            data = tool.model_dump()
            assert data["name"] == tool.name


class TestToolExecution:
    def test_list_files(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "list_files", {})
        assert "doc.md" in result
        assert "faq.txt" in result

    def test_read_file(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "read_file", {"path": "doc.md"})
        assert "Title" in result

    def test_search_in_file(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "search_in_file", {"path": "doc.md", "pattern": "refund"})
        assert "refund" in result

    def test_grep_corpus(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "grep_corpus", {"pattern": "refund"})
        assert "doc.md" in result
        assert "faq.txt" in result

    def test_file_info(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "file_info", {"path": "doc.md"})
        assert "lines" in result
        assert "bytes" in result

    def test_corpus_tree(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "corpus_tree", {})
        assert "doc.md" in result

    def test_unknown_tool(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_file_not_found_error(self, corpus: Corpus) -> None:
        result = execute_tool(corpus, "read_file", {"path": "missing.md"})
        assert "Error" in result
