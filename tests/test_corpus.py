"""Tests for the Corpus class."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_doc_search.corpus import Corpus


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Corpus:
    """Create a small test corpus."""
    # Create some test files
    (tmp_path / "readme.md").write_text("# My Project\n\nThis is the readme.\n")
    (tmp_path / "notes.txt").write_text("Some notes here.\nLine 2.\nLine 3.\n")

    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "refund.md").write_text(
        "# Refund Policy\n\n"
        "You can return items within 30 days.\n"
        "Refunds are processed in 5-7 business days.\n"
    )
    (policies / "shipping.md").write_text(
        "# Shipping Policy\n\n"
        "Free shipping on orders over $50.\n"
        "Standard shipping takes 3-5 days.\n"
    )

    # A file that should be ignored (binary-ish extension)
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    # A directory that should be ignored
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("gitconfig")

    return Corpus(path=tmp_path)


class TestCorpusDiscovery:
    def test_discover_finds_text_files(self, sample_corpus: Corpus) -> None:
        files = sample_corpus.discover_files()
        assert "readme.md" in files
        assert "notes.txt" in files
        assert "policies/refund.md" in files
        assert "policies/shipping.md" in files

    def test_discover_ignores_binary_files(self, sample_corpus: Corpus) -> None:
        files = sample_corpus.discover_files()
        assert "image.png" not in files

    def test_discover_ignores_git_dir(self, sample_corpus: Corpus) -> None:
        files = sample_corpus.discover_files()
        assert not any(".git" in f for f in files)

    def test_discover_returns_sorted(self, sample_corpus: Corpus) -> None:
        files = sample_corpus.discover_files()
        assert files == sorted(files)


class TestCorpusRead:
    def test_read_full_file(self, sample_corpus: Corpus) -> None:
        content = sample_corpus.read_file("readme.md")
        assert "1: # My Project" in content
        assert "3: This is the readme." in content

    def test_read_line_range(self, sample_corpus: Corpus) -> None:
        content = sample_corpus.read_file("notes.txt", start_line=2, end_line=3)
        assert "2: Line 2." in content
        assert "3: Line 3." in content
        assert "1:" not in content

    def test_read_nonexistent_file_raises(self, sample_corpus: Corpus) -> None:
        with pytest.raises(FileNotFoundError):
            sample_corpus.read_file("nonexistent.md")


class TestCorpusSearch:
    def test_search_in_file_finds_matches(self, sample_corpus: Corpus) -> None:
        result = sample_corpus.search_in_file("policies/refund.md", "30 days")
        assert "30 days" in result

    def test_search_in_file_no_matches(self, sample_corpus: Corpus) -> None:
        result = sample_corpus.search_in_file("policies/refund.md", "cryptocurrency")
        assert "No matches" in result

    def test_grep_across_corpus(self, sample_corpus: Corpus) -> None:
        result = sample_corpus.grep("shipping")
        assert "policies/shipping.md" in result

    def test_grep_no_results(self, sample_corpus: Corpus) -> None:
        result = sample_corpus.grep("xyznonexistent")
        assert "No matches" in result


class TestCorpusTree:
    def test_tree_output(self, sample_corpus: Corpus) -> None:
        tree = sample_corpus.tree()
        assert "policies/" in tree
        assert "readme.md" in tree

    def test_tree_excludes_git(self, sample_corpus: Corpus) -> None:
        tree = sample_corpus.tree()
        assert ".git" not in tree


class TestCorpusValidation:
    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            Corpus(path=tmp_path / "nonexistent")

    def test_path_escape_raises(self, sample_corpus: Corpus) -> None:
        with pytest.raises((FileNotFoundError, PermissionError)):
            sample_corpus.read_file("../../etc/passwd")

    def test_custom_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n")
        (tmp_path / "readme.md").write_text("hello\n")
        corpus = Corpus(path=tmp_path, extensions=frozenset({".csv"}))
        files = corpus.discover_files()
        assert "data.csv" in files
        assert "readme.md" not in files

    def test_model_serialization(self, sample_corpus: Corpus) -> None:
        """Corpus should be serializable via Pydantic."""
        data = sample_corpus.model_dump()
        assert "path" in data
        assert "extensions" in data
