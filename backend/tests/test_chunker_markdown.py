"""Tests for chunking self-authored internal corpus documents.

Internal documents go through the same NLI gate as published literature, so they
need the same cleanup: markup that survives into a chunk is noise in a retrieval
target and worse in an entailment premise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunker import chunk_markdown, strip_markdown_inline

LONG = (
    "Depth is the primary determinant of management because lesions reaching the "
    "subchondral bone do not heal spontaneously, whereas superficial fibrillation "
    "of the articular surface frequently remains stable over long periods of "
    "clinical follow-up in the absence of mechanical symptoms. "
)


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "cartilage-notes.md"
    path.write_text(
        "# Cartilage lesion reference\n\n"
        "Intro paragraph that is long enough to survive the length filter. " + LONG +
        "\n\n"
        "## Depth grading\n\n"
        f"**Grade 3** is a *full-thickness* defect with `subchondral` bone exposed. "
        f"{LONG}\n\n"
        "## Acknowledgements\n\n"
        "Thanks to colleagues who reviewed this document and offered comments on "
        "the clinical accuracy of the grading descriptions presented here.\n",
        encoding="utf-8",
    )
    return path


# ── Inline markup ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**Osteophytes** are bony", "Osteophytes are bony"),
        ("a *full-thickness* defect", "a full-thickness defect"),
        ("the `subchondral` bone", "the subchondral bone"),
        ("__strong__ and _weak_", "strong and weak"),
        ("see [the guideline](https://example.org/x)", "see the guideline"),
        ("plain text unchanged", "plain text unchanged"),
    ],
)
def test_strip_markdown_inline(raw: str, expected: str) -> None:
    assert strip_markdown_inline(raw) == expected


def test_chunks_contain_no_markdown_markers(doc: Path) -> None:
    for chunk in chunk_markdown(doc):
        assert "**" not in chunk.text
        assert "`" not in chunk.text
        assert "](" not in chunk.text


# ── Structure and provenance ──────────────────────────────────────────────────


def test_heading_becomes_the_section_path(doc: Path) -> None:
    paths = {c.section_path for c in chunk_markdown(doc)}
    assert "Depth grading" in paths


def test_document_title_comes_from_the_h1(doc: Path) -> None:
    chunk = chunk_markdown(doc)[0]
    assert chunk.title == "Cartilage lesion reference"
    assert chunk.text.startswith("Cartilage lesion reference")


def test_internal_documents_are_tagged_by_provenance(doc: Path) -> None:
    """So the evaluation can report how much grounding was self-authored."""
    for chunk in chunk_markdown(doc):
        assert chunk.source_type == "internal"
        assert chunk.source_id == "cartilage-notes"
        assert chunk.url is None


def test_administrative_headings_are_dropped(doc: Path) -> None:
    joined = " ".join(c.text for c in chunk_markdown(doc))
    assert "Thanks to colleagues" not in joined


def test_chunk_ids_are_deterministic(doc: Path) -> None:
    assert [c.chunk_id for c in chunk_markdown(doc)] == [
        c.chunk_id for c in chunk_markdown(doc)
    ]
