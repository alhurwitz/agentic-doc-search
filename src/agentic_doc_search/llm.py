"""LLM abstraction layer — provider-agnostic via litellm.

Configuration is handled via pydantic-settings, so values can come from
constructor arguments, environment variables, or .env files.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_doc_search.models import TokenUsage
from agentic_doc_search.tools import ToolDefinition

logger = logging.getLogger(__name__)

# Suppress litellm's noisy logging by default
litellm.suppress_debug_info = True


class LLMConfig(BaseSettings):
    """Configuration for the LLM backend.

    Values can be set via constructor, environment variables (prefixed with ADS_),
    or a .env file.

    Environment variable examples:
        ADS_MODEL=gpt-4o
        ADS_API_KEY=sk-...
        ADS_API_BASE=http://localhost:11434
        ADS_TEMPERATURE=0.0
        ADS_MAX_TOKENS=4096
    """

    model_config = SettingsConfigDict(
        env_prefix="ADS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = "claude-sonnet-4-20250514"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: SecretStr | None = None
    api_base: str | None = None
    max_retries: int = Field(default=3, ge=0)


# Transient errors worth retrying
_RETRYABLE_ERRORS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
)


def chat_completion(
    config: LLMConfig,
    messages: list[dict[str, Any]],
    tools: list[ToolDefinition] | None = None,
) -> tuple[Any, TokenUsage]:
    """Send a chat completion request via litellm with automatic retries.

    Args:
        config: LLM configuration.
        messages: Conversation messages in OpenAI format.
        tools: Optional tool definitions for function calling.

    Returns:
        Tuple of (response message, token usage).
    """

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _call() -> Any:
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        if config.api_key:
            kwargs["api_key"] = config.api_key.get_secret_value()
        if config.api_base:
            kwargs["api_base"] = config.api_base

        if tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]

        return litellm.completion(**kwargs)

    response = _call()

    usage = TokenUsage()
    if hasattr(response, "usage") and response.usage:
        usage.prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        usage.completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

    return response.choices[0].message, usage
