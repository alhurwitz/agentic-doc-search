"""Corpus — a lightweight wrapper around a directory of documents."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator


# Default file extensions treated as text documents
DEFAULT_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".txt", ".html", ".htm", ".rst", ".org",
    ".csv", ".json", ".yaml", ".yml", ".toml", ".xml",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".rb", ".sh", ".css", ".sql", ".r", ".R", ".log",
})

# Directories to always skip
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", ".tox", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", "dist", "build", ".eggs",
})


class Corpus(BaseModel):
    """A searchable collection of text documents in a directory.

    The corpus does no indexing — it provides file discovery and basic
    text operations that the agent uses as tools.

    Args:
        path: Root directory of the document corpus.
        extensions: File extensions to include. Defaults to common text formats.
        ignore_dirs: Directory names to skip during traversal.
        include_patterns: Optional glob patterns to include (overrides extensions).
        exclude_patterns: Optional glob patterns to exclude.
    """

    model_config = {"arbitrary_types_allowed": True}

    path: Path
    extensions: frozenset[str] = DEFAULT_EXTENSIONS
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)

    # Class-level constants
    _default_extensions: ClassVar[frozenset[str]] = DEFAULT_EXTENSIONS
    _default_ignore_dirs: ClassVar[frozenset[str]] = DEFAULT_IGNORE_DIRS

    @field_validator("path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @model_validator(mode="after")
    def validate_path_is_directory(self) -> "Corpus":
        if not self.path.is_dir():
            raise ValueError(f"Corpus path is not a directory: {self.path}")
        return self

    def discover_files(self) -> list[str]:
        """Return all document file paths relative to the corpus root."""
        files: list[str] = []
        for root, dirs, filenames in os.walk(self.path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for fname in sorted(filenames):
                rel_path = os.path.relpath(os.path.join(root, fname), self.path)

                # Check exclude patterns
                if any(fnmatch.fnmatch(rel_path, pat) for pat in self.exclude_patterns):
                    continue

                # Check include patterns (if set, they override extension filtering)
                if self.include_patterns:
                    if any(fnmatch.fnmatch(rel_path, pat) for pat in self.include_patterns):
                        files.append(rel_path)
                else:
                    # Fall back to extension filtering
                    _, ext = os.path.splitext(fname)
                    if ext.lower() in self.extensions:
                        files.append(rel_path)

        return sorted(files)

    def read_file(self, rel_path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file's contents, optionally slicing by line numbers.

        Args:
            rel_path: Path relative to corpus root.
            start_line: First line to return (1-indexed, inclusive).
            end_line: Last line to return (1-indexed, inclusive). None means EOF.

        Returns:
            The file content (or slice) as a string with line numbers prefixed.
        """
        full_path = self.path / rel_path
        self._validate_file_path(full_path)

        text = full_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # Convert to 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line is not None else len(lines)

        selected = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1}: {line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)

    def search_in_file(self, rel_path: str, pattern: str) -> str:
        """Search for a regex pattern within a single file.

        Returns matching lines with line numbers.
        """
        full_path = self.path / rel_path
        self._validate_file_path(full_path)

        text = full_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        regex = re.compile(pattern, re.IGNORECASE)

        matches: list[str] = []
        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{i}: {line}")

        if not matches:
            return f"No matches for '{pattern}' in {rel_path}"
        return "\n".join(matches)

    def grep(self, pattern: str, max_results: int = 50) -> str:
        """Search for a regex pattern across the entire corpus.

        Returns matching file:line entries, up to max_results.
        """
        regex = re.compile(pattern, re.IGNORECASE)
        results: list[str] = []

        for rel_path in self.discover_files():
            full_path = self.path / rel_path
            try:
                text = full_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel_path}:{i}: {line.strip()}")
                    if len(results) >= max_results:
                        return "\n".join(results) + f"\n... (truncated at {max_results} results)"

        if not results:
            return f"No matches for '{pattern}' across the corpus"
        return "\n".join(results)

    def file_info(self, rel_path: str) -> str:
        """Get basic info about a file (size, line count)."""
        full_path = self.path / rel_path
        self._validate_file_path(full_path)

        stat = full_path.stat()
        text = full_path.read_text(encoding="utf-8", errors="replace")
        line_count = len(text.splitlines())

        return f"{rel_path}: {line_count} lines, {stat.st_size} bytes"

    def tree(self, max_depth: int = 3) -> str:
        """Return a tree-like listing of the corpus directory structure."""
        lines: list[str] = [f"{self.path.name}/"]
        self._tree_walk(self.path, "", max_depth, 0, lines)
        return "\n".join(lines)

    def _tree_walk(
        self, dir_path: Path, prefix: str, max_depth: int, depth: int, lines: list[str]
    ) -> None:
        if depth >= max_depth:
            return

        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        entries = [
            e for e in entries if not (e.is_dir() and e.name in self.ignore_dirs)
        ]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                self._tree_walk(entry, prefix + extension, max_depth, depth + 1, lines)

    def _validate_file_path(self, full_path: Path) -> None:
        """Ensure a path exists and is within the corpus root."""
        resolved = full_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {full_path}")
        if not str(resolved).startswith(str(self.path)):
            raise PermissionError(f"Path escapes corpus root: {full_path}")
