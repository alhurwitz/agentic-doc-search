# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`agentic-doc-search` is a vectorless agentic document search system. Instead of using embeddings and vector databases, it employs an LLM agent with file-exploration tools to navigate and reason through document corpora — similar to how a human researcher would work.

The agent uses a **ReAct (Reason + Act) loop** where it observes the corpus structure, searches for keywords, reads relevant files, refines its search, and synthesizes an answer with citations.

## Development Setup

This project uses **uv** for Python package management.

```bash
# Install dependencies (including dev dependencies)
uv sync --dev

# Run the CLI locally
uv run ads search ./my-docs "What is the refund policy?"
```

## Common Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run tests verbosely
uv run pytest -v

# Run a single test file
uv run pytest tests/test_corpus.py

# Run a specific test
uv run pytest tests/test_corpus.py::test_discover_files
```

### Code Quality
```bash
# Lint code
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

### CLI Usage
```bash
# Search a corpus
uv run ads search ./my-docs "query here"

# Use a different model
uv run ads search ./my-docs "query" -m gpt-4o

# Show corpus structure
uv run ads tree ./my-docs

# List all document files
uv run ads files ./my-docs

# Verbose mode (shows tool call arguments)
uv run ads search ./my-docs "query" -v
```

## Architecture

### Core Components

- **`agent.py`** - The ReAct agent that orchestrates the search loop
  - Maintains conversation history with the LLM
  - Iteratively calls tools based on LLM decisions
  - Tracks token usage and reasoning steps
  - System prompt at line 17 defines the agent's search strategy

- **`corpus.py`** - Manages document discovery and text operations
  - No indexing or preprocessing — on-demand file reading
  - File discovery with extension filtering and ignore patterns
  - Line-numbered file reading for precise citations
  - Regex-based search (case-insensitive)

- **`tools.py`** - Defines 6 tools available to the agent:
  1. `list_files` - List documents in corpus or subdirectory
  2. `read_file` - Read file contents with optional line ranges
  3. `search_in_file` - Search for pattern within a specific file
  4. `grep_corpus` - Search pattern across all files
  5. `file_info` - Get file metadata (line count, size)
  6. `corpus_tree` - Display directory structure

- **`llm.py`** - Provider-agnostic LLM abstraction via litellm
  - Supports any model litellm supports (Claude, GPT, Ollama, etc.)
  - Function-calling tool support (OpenAI format)
  - Token usage tracking

- **`models.py`** - Data models:
  - `SearchResult` - Final answer with sources, reasoning trace, token usage
  - `ToolCall` - Record of tool invocations
  - `SourceReference` - File location references
  - `TokenUsage` - Token consumption metrics

- **`cli.py`** - Click-based CLI with rich formatting
  - Commands: `search`, `tree`, `files`
  - Supports model selection, API key override, verbose mode

### Key Design Patterns

1. **No Preprocessing**: Documents aren't indexed or embedded upfront. The corpus is explored dynamically during each query.

2. **Tool Execution Flow**:
   - Agent asks LLM for next action
   - LLM returns tool calls (OpenAI function-calling format)
   - Tools execute against Corpus
   - Results added to conversation history
   - Loop continues until LLM provides final answer or max iterations reached

3. **Line Numbering**: All file reads return line-numbered text (format: `{line_num}: {content}`) to enable precise citations.

4. **Lazy Agent Import**: `__init__.py` uses `__getattr__` for lazy import of `Agent` to reduce startup time.

## Model Support

Supports any model that litellm supports:
- **Anthropic**: `claude-sonnet-4-20250514`, `claude-opus-4-20250514`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Local**: `ollama/llama3` (via Ollama)
- **Cloud providers**: Azure, AWS Bedrock, Google Vertex

Default model: `claude-sonnet-4-20250514`

## File Format Support

Currently supports text-based formats (see `DEFAULT_EXTENSIONS` in `corpus.py`):
- Documentation: `.md`, `.txt`, `.rst`, `.html`
- Config: `.json`, `.yaml`, `.toml`, `.xml`, `.csv`
- Code: `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, `.sh`, `.sql`

Binary formats (PDF, DOCX) are planned but not yet implemented.

## Important Implementation Details

- **Agent System Prompt**: The strategy defined in `agent.py:17-35` guides how the agent approaches search tasks. Modifications here significantly impact agent behavior.

- **Max Iterations**: Default 20 iterations (`agent.py:37`). If reached, agent is forced to provide answer based on current findings.

- **Corpus Path Validation**: `corpus.py:237-243` ensures paths stay within corpus root to prevent directory traversal attacks.

- **Error Handling in Tools**: Tool execution catches `FileNotFoundError`, `PermissionError`, and generic exceptions, returning error messages as strings to the LLM.

- **Case-Insensitive Search**: All regex searches in `corpus.py` use `re.IGNORECASE`.

- **Result Truncation**: `grep_corpus` truncates at `max_results` (default 50) to avoid overwhelming context.

## Testing Notes

- Tests use pytest with async support enabled (`asyncio_mode = "auto"`)
- Test files are in `tests/` and mirror the source structure
- No test fixtures or mocks observed yet — likely testing real file operations
