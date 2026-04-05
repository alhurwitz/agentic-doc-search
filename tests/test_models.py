"""Tests for Pydantic data models."""

from __future__ import annotations

from agentic_doc_search.models import SearchResult, SourceReference, TokenUsage, ToolCall


class TestSourceReference:
    def test_str_file_only(self) -> None:
        ref = SourceReference(file_path="docs/readme.md")
        assert str(ref) == "docs/readme.md"

    def test_str_with_line_start(self) -> None:
        ref = SourceReference(file_path="docs/readme.md", line_start=10)
        assert str(ref) == "docs/readme.md:10"

    def test_str_with_line_range(self) -> None:
        ref = SourceReference(file_path="docs/readme.md", line_start=10, line_end=20)
        assert str(ref) == "docs/readme.md:10-20"

    def test_str_same_start_end(self) -> None:
        ref = SourceReference(file_path="docs/readme.md", line_start=10, line_end=10)
        assert str(ref) == "docs/readme.md:10"

    def test_serialization(self) -> None:
        ref = SourceReference(file_path="test.md", line_start=1, snippet="hello")
        data = ref.model_dump()
        assert data["file_path"] == "test.md"
        assert data["snippet"] == "hello"
        restored = SourceReference.model_validate(data)
        assert restored == ref


class TestTokenUsage:
    def test_total_tokens(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestToolCall:
    def test_creation(self) -> None:
        tc = ToolCall(tool_name="grep_corpus", arguments={"pattern": "refund"}, result="found it")
        assert tc.tool_name == "grep_corpus"
        assert tc.arguments == {"pattern": "refund"}

    def test_serialization_roundtrip(self) -> None:
        tc = ToolCall(tool_name="read_file", arguments={"path": "doc.md"}, result="contents")
        data = tc.model_dump()
        restored = ToolCall.model_validate(data)
        assert restored == tc


class TestSearchResult:
    def test_full_result(self) -> None:
        result = SearchResult(
            query="What is the refund policy?",
            answer="You can return items within 30 days.",
            sources=[
                SourceReference(file_path="policies/refund.md", line_start=3, line_end=4),
            ],
            reasoning=[
                ToolCall(tool_name="grep_corpus", arguments={"pattern": "refund"}, result="..."),
                ToolCall(tool_name="read_file", arguments={"path": "policies/refund.md"}, result="..."),
            ],
            token_usage=TokenUsage(prompt_tokens=500, completion_tokens=100),
        )
        assert result.token_usage.total_tokens == 600
        assert len(result.sources) == 1
        assert len(result.reasoning) == 2

    def test_minimal_result(self) -> None:
        result = SearchResult(query="test", answer="answer")
        assert result.sources == []
        assert result.reasoning == []
        assert result.token_usage.total_tokens == 0

    def test_serialization(self) -> None:
        result = SearchResult(
            query="test",
            answer="answer",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        data = result.model_dump()
        assert data["query"] == "test"
        assert data["token_usage"]["prompt_tokens"] == 10
