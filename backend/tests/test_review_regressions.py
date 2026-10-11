"""Regressions for bugs found in the Phase 0 code review.

Each test here failed when written. They are grouped separately from the
feature tests so the review trail stays visible.
"""

from __future__ import annotations

import pytest

from rag.chunker import chunk_jats
from rag.corpus_types import ManifestEntry
from rag.licensing import LicenceNotRedistributable, assert_commercial_licence

LONG = (
    "Radiographic assessment proceeds by measuring the medial and lateral joint "
    "space width on a weight-bearing view, and osteophyte formation at the tibial "
    "plateau margins is graded separately from joint space narrowing because the "
    "two features progress at different rates in longitudinal cohorts. "
)

CC_BY = (
    "<permissions><license><ali:license_ref>"
    "https://creativecommons.org/licenses/by/4.0/"
    "</ali:license_ref></license></permissions>"
)


def _manifest(title: str = "Knee grading") -> ManifestEntry:
    return ManifestEntry(
        pmcid="PMC1",
        title=title,
        licence_ref="https://creativecommons.org/licenses/by/4.0/",
        licence_source="ali:license_ref",
        sha256="z",
        source_url="u",
        retrieved_at="t",
    )


def _article(
    title: str, abstract: str = "", body: str = "", perms: str = CC_BY
) -> bytes:
    return f"""<?xml version="1.0"?>
<article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
         xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>{title}</article-title></title-group>
    {perms}
    {abstract}
  </article-meta></front>
  <body>{body}</body>
</article>""".encode()


# ── Chunker ───────────────────────────────────────────────────────────────────


def test_abstract_is_not_chunked_twice() -> None:
    """The abstract's paragraphs were collected both directly and via the walker."""
    xml = _article(
        "Knee grading",
        abstract=f"<abstract><p>{LONG}</p></abstract>",
        body=f"<sec><title>Results</title><p>{LONG}</p></sec>",
    )
    chunks = chunk_jats(xml, _manifest())
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts)), "duplicate chunk text emitted"
    assert sum(1 for c in chunks if c.section_path == "Abstract") == 1


def test_structured_abstract_subsections_are_kept_once() -> None:
    xml = _article(
        "Knee grading",
        abstract=(
            "<abstract>"
            f"<sec><title>Background</title><p>{LONG}</p></sec>"
            f"<sec><title>Results</title><p>{LONG}2</p></sec>"
            "</abstract>"
        ),
    )
    chunks = chunk_jats(xml, _manifest())
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))
    paths = sorted(c.section_path for c in chunks)
    assert paths == ["Abstract > Background", "Abstract > Results"]


def test_quality_filter_measures_the_body_not_a_mis_split_header() -> None:
    """A title containing ": " used to leave header remnants in the measured body.

    The body here is just under the minimum length; with header text leaking into
    the measurement the chunk would be wrongly retained.
    """
    short_body = "Joint space was measured on both sides of the knee in all cases."
    xml = _article(
        "Knee osteoarthritis: a structured review of radiographic grading methods",
        body=f"<sec><title>Results</title><p>{short_body}</p></sec>",
    )
    assert chunk_jats(xml, _manifest()) == []


def test_consecutive_chunks_overlap_by_one_paragraph() -> None:
    """Overlap keeps content that straddles a chunk boundary retrievable.

    Each paragraph carries a unique marker, so overlap is detected by the marker
    itself rather than by filler text that would repeat anyway.
    """
    markers = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]
    body = "<sec><title>Discussion</title>" + "".join(
        f"<p>Marker {m}. {LONG}</p>" for m in markers
    ) + "</sec>"
    chunks = chunk_jats(_article("Knee grading", body=body), _manifest())
    assert len(chunks) >= 2

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        shared = {m for m in markers if m in earlier.text} & {
            m for m in markers if m in later.text
        }
        assert shared, (
            f"no shared paragraph between consecutive chunks "
            f"{earlier.chunk_id} and {later.chunk_id}"
        )


# ── Licence gate ──────────────────────────────────────────────────────────────


def test_mixed_licences_in_permissions_are_refused() -> None:
    """One allowed and one disallowed declaration must refuse, not pass."""
    mixed = (
        "<permissions><license>"
        "<ali:license_ref>https://creativecommons.org/licenses/by/4.0/</ali:license_ref>"
        "</license><license>"
        "<ali:license_ref>https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref>"
        "</license></permissions>"
    )
    xml = _article("Knee grading", perms=mixed)
    with pytest.raises(LicenceNotRedistributable, match="conflicting"):
        assert_commercial_licence("PMC1", xml)


def test_no_derivatives_bucket_is_not_accepted() -> None:
    """ND is outside our policy even though PMC files it under commercial use.

    The corpus ships chunked excerpts with prepended context headers, which is a
    derivative work.
    """
    from rag.licensing import ALLOWED_OA_BUCKETS

    assert "cc by-nd" not in ALLOWED_OA_BUCKETS
    xml = _article("Knee grading")
    with pytest.raises(LicenceNotRedistributable):
        assert_commercial_licence("PMC1", xml, oa_bucket="CC BY-ND")


# ── Readiness reporting ───────────────────────────────────────────────────────


def test_index_is_not_ready_when_only_a_stale_lock_file_exists(
    tmp_path, monkeypatch
) -> None:
    """A leftover lock from a failed ingest must not report the index as ready."""
    from rag import config

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / ".lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    assert config.rag_index_ready() is False

    (index_dir / "collection").mkdir()
    assert config.rag_index_ready() is True
