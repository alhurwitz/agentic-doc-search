"""Data models for search results and tool interactions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A reference to a specific location in a document."""

    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str = ""

    def __str__(self) -> str:
        loc = self.file_path
        if self.line_start is not None:
            loc += f":{self.line_start}"
            if self.line_end is not None and self.line_end != self.line_start:
                loc += f"-{self.line_end}"
        return loc


class ToolCall(BaseModel):
    """A record of a single tool invocation by the agent."""

    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: str = ""


class TokenUsage(BaseModel):
    """Token consumption for a search."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SearchResult(BaseModel):
    """The result of an agentic search over a corpus."""

    model_config = {"arbitrary_types_allowed": True}

    query: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    reasoning: list[ToolCall] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
