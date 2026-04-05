"""Agent tools — the functions the LLM can call to navigate the corpus.

Each tool is a Pydantic model with:
- A name and description (sent to the LLM as the tool schema)
- A parameter schema (JSON Schema for the LLM to fill in)
- Execution handled by execute_tool()
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentic_doc_search.corpus import Corpus


class ToolDefinition(BaseModel):
    """Schema for a tool that the LLM can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to the OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool definitions (schemas sent to the LLM)
# ---------------------------------------------------------------------------

LIST_FILES = ToolDefinition(
    name="list_files",
    description=(
        "List all document files in the corpus, or files within a specific subdirectory. "
        "Returns file paths relative to the corpus root. Use this to understand what "
        "documents are available before reading them."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subdirectory": {
                "type": "string",
                "description": (
                    "Optional subdirectory to list (relative to corpus root). "
                    "Omit to list all files in the entire corpus."
                ),
            },
        },
        "required": [],
    },
)

READ_FILE = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a document file. Returns the text with line numbers. "
        "For large files, use start_line and end_line to read specific sections."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the corpus root.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed). Defaults to 1.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-indexed, inclusive). Omit to read to end of file.",
            },
        },
        "required": ["path"],
    },
)

SEARCH_IN_FILE = ToolDefinition(
    name="search_in_file",
    description=(
        "Search for a pattern within a specific file. Returns matching lines with "
        "line numbers. The pattern is a case-insensitive regular expression."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the corpus root.",
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for (case-insensitive).",
            },
        },
        "required": ["path", "pattern"],
    },
)

GREP_CORPUS = ToolDefinition(
    name="grep_corpus",
    description=(
        "Search for a pattern across ALL files in the corpus. Returns matching "
        "file:line entries. Use this to find which files contain relevant information "
        "before reading them in detail. The pattern is a case-insensitive regex."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for across all files (case-insensitive).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 50.",
            },
        },
        "required": ["pattern"],
    },
)

FILE_INFO = ToolDefinition(
    name="file_info",
    description=(
        "Get basic metadata about a file — line count and size in bytes. "
        "Useful to check how large a file is before deciding to read it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the corpus root.",
            },
        },
        "required": ["path"],
    },
)

CORPUS_TREE = ToolDefinition(
    name="corpus_tree",
    description=(
        "Show the directory structure of the corpus as a tree. "
        "Useful to understand how documents are organized before searching."
    ),
    parameters={
        "type": "object",
        "properties": {
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth to display. Defaults to 3.",
            },
        },
        "required": [],
    },
)

# All available tools
ALL_TOOLS: list[ToolDefinition] = [
    LIST_FILES,
    READ_FILE,
    SEARCH_IN_FILE,
    GREP_CORPUS,
    FILE_INFO,
    CORPUS_TREE,
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def execute_tool(corpus: Corpus, tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool call against a corpus and return the result as a string."""
    try:
        if tool_name == "list_files":
            subdir = arguments.get("subdirectory")
            files = corpus.discover_files()
            if subdir:
                files = [f for f in files if f.startswith(subdir)]
            if not files:
                return "No files found."
            return "\n".join(files)

        elif tool_name == "read_file":
            return corpus.read_file(
                arguments["path"],
                start_line=arguments.get("start_line", 1),
                end_line=arguments.get("end_line"),
            )

        elif tool_name == "search_in_file":
            return corpus.search_in_file(arguments["path"], arguments["pattern"])

        elif tool_name == "grep_corpus":
            return corpus.grep(
                arguments["pattern"],
                max_results=arguments.get("max_results", 50),
            )

        elif tool_name == "file_info":
            return corpus.file_info(arguments["path"])

        elif tool_name == "corpus_tree":
            return corpus.tree(max_depth=arguments.get("max_depth", 3))

        else:
            return f"Unknown tool: {tool_name}"

    except FileNotFoundError as e:
        return f"Error: {e}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error executing {tool_name}: {type(e).__name__}: {e}"
