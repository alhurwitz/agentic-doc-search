"""The core agent — a ReAct loop that navigates a corpus to answer questions."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pydantic import BaseModel, Field

from agentic_doc_search.corpus import Corpus
from agentic_doc_search.llm import LLMConfig, chat_completion
from agentic_doc_search.models import SearchResult, ToolCall, TokenUsage
from agentic_doc_search.tools import ALL_TOOLS, ToolDefinition, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research agent that answers questions by exploring a document corpus.

You have access to tools that let you navigate, read, and search through files.
Use them iteratively to find the information needed to answer the user's question.

Strategy:
1. Start by understanding the corpus structure (corpus_tree or list_files).
2. Search broadly for relevant keywords (grep_corpus).
3. Read promising files in detail (read_file).
4. Search within specific files for targeted information (search_in_file).
5. When you have enough information, provide a comprehensive answer.

Rules:
- Always cite your sources with file paths and line numbers.
- If you cannot find the answer, say so — do not make things up.
- Be thorough but efficient — don't read every file if a grep narrows it down.
- Think step by step about what to search for next.
"""

DEFAULT_MAX_ITERATIONS = 20


class Agent(BaseModel):
    """An agentic search agent that explores a corpus to answer questions.

    Args:
        model: Model identifier (anything litellm supports).
        llm_config: Full LLM configuration. If provided, `model` is ignored.
        system_prompt: Custom system prompt. Defaults to the built-in research prompt.
        max_iterations: Maximum tool-calling iterations before forcing an answer.
        tools: Custom tool definitions. Defaults to all built-in tools.
        on_tool_call: Optional callback invoked after each tool call, for logging/streaming.
    """

    model_config = {"arbitrary_types_allowed": True}

    model: str = "claude-sonnet-4-20250514"
    llm_config: LLMConfig | None = None
    system_prompt: str = SYSTEM_PROMPT
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, gt=0)
    tools: list[ToolDefinition] = Field(default_factory=lambda: list(ALL_TOOLS))
    on_tool_call: Callable[[ToolCall], None] | None = Field(default=None, exclude=True)

    def _get_config(self) -> LLMConfig:
        if self.llm_config is not None:
            return self.llm_config
        return LLMConfig(model=self.model)

    def search(self, corpus: Corpus, query: str) -> SearchResult:
        """Search the corpus for an answer to the given query.

        This runs a ReAct loop: the LLM reasons about the query, picks tools
        to explore the corpus, observes results, and iterates until it has
        enough to synthesize an answer.

        Args:
            corpus: The document corpus to search.
            query: The user's question.

        Returns:
            A SearchResult with the answer, sources, and reasoning trace.
        """
        config = self._get_config()
        total_usage = TokenUsage()
        tool_calls_log: list[ToolCall] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        for iteration in range(self.max_iterations):
            logger.debug(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            response_msg, usage = chat_completion(config, messages, tools=self.tools)
            total_usage.prompt_tokens += usage.prompt_tokens
            total_usage.completion_tokens += usage.completion_tokens

            # Check if the LLM wants to call tools
            tool_calls_raw = getattr(response_msg, "tool_calls", None)

            if not tool_calls_raw:
                # No tool calls — the agent is done, this is the final answer
                answer = getattr(response_msg, "content", "") or ""
                return SearchResult(
                    query=query,
                    answer=answer,
                    sources=[],  # TODO: parse source references from the answer
                    reasoning=tool_calls_log,
                    token_usage=total_usage,
                )

            # Process each tool call
            # Add the assistant message with tool calls to the conversation
            messages.append(_message_to_dict(response_msg))

            for tc in tool_calls_raw:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                logger.debug(f"Tool call: {func_name}({func_args})")

                result_str = execute_tool(corpus, func_name, func_args)

                # Log the tool call
                tool_call_record = ToolCall(
                    tool_name=func_name,
                    arguments=func_args,
                    result=result_str,
                )
                tool_calls_log.append(tool_call_record)

                if self.on_tool_call:
                    self.on_tool_call(tool_call_record)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

        # Max iterations reached — ask for a final answer without tools
        messages.append(
            {
                "role": "user",
                "content": (
                    "You've reached the maximum number of search steps. "
                    "Please provide your best answer based on what you've found so far."
                ),
            }
        )
        response_msg, usage = chat_completion(config, messages, tools=None)
        total_usage.prompt_tokens += usage.prompt_tokens
        total_usage.completion_tokens += usage.completion_tokens

        answer = getattr(response_msg, "content", "") or ""
        return SearchResult(
            query=query,
            answer=answer,
            sources=[],
            reasoning=tool_calls_log,
            token_usage=total_usage,
        )


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a litellm response message object to a dict for the messages list."""
    d: dict[str, Any] = {"role": "assistant"}

    content = getattr(msg, "content", None)
    if content:
        d["content"] = content

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    return d
