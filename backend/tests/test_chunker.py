"""Tests for JATS chunking.

The chunker is the highest-leverage code in the corpus pipeline: a bad chunk is
both an unretrievable passage and a useless entailment premise. These tests pin
down the specific cleanups that make a research-article paragraph usable, each of
which was a deliberate decision rather than an incidental detail.
"""

from __future__ import annotations

import pytest

from rag.chunker import chunk_jats, section_is_dropped
from rag.corpus_types import ManifestEntry

MANIFEST = ManifestEntry(
    pmcid="PMC123456",
    title="Radiographic grading of knee osteoarthritis",
    licence_ref="https://creativecommons.org/licenses/by/4.0/",
    licence_source="ali:license_ref",
    sha256="deadbeef",
    source_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/",
    retrieved_at="2026-10-11T00:00:00Z",
    journal="BMC Musculoskeletal Disorders",
    year="2021",
)

LONG_A = (
    "Radiographic assessment of the tibiofemoral joint proceeds by measuring the "
    "medial and lateral joint space width on a weight-bearing anteroposterior view. "
    "Osteophyte formation at the tibial plateau margins is graded separately from "
    "joint space narrowing, because the two features progress at different rates in "
    "most longitudinal cohorts described in the literature to date. "
)
LONG_B = (
    "Subchondral sclerosis appears later than osteophytosis and is most reliably "
    "identified in the medial compartment of the weight-bearing surface. Observers "
    "differ most at the boundary between doubtful and minimal disease, which is the "
    "transition where automated grading systems also lose the greatest accuracy. "
)
LONG_C = (
    "Cartilage loss is not directly visible on plain radiography and must be inferred "
    "from the width of the remaining joint space, which is why magnetic resonance "
    "imaging remains necessary whenever soft-tissue pathology is clinically suspected "
    "in a patient presenting with mechanical symptoms of the knee joint. "
)


def _article(body: str, back: str = "", permissions: str | None = None) -> bytes:
    perms = (
        permissions
        if permissions is not None
        else """<permissions><license>
          <ali:license_ref>https://creativecommons.org/licenses/by/4.0/</ali:license_ref>
        </license></permissions>"""
    )
    return f"""<?xml version="1.0"?>
<article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
         xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>Radiographic grading of knee
      osteoarthritis</article-title></title-group>
    {perms}
  </article-meta></front>
  <body>{body}</body>
  <back>{back}</back>
</article>""".encode()


# ── Structure ─────────────────────────────────────────────────────────────────


def test_chunks_carry_a_context_header() -> None:
    """A passage must name its article and section, or it is unusable alone."""
    xml = _article(f"<sec><title>Results</title><p>{LONG_A}</p></sec>")
    chunks = chunk_jats(xml, MANIFEST)
    assert chunks
    assert chunks[0].text.startswith(
        "Radiographic grading of knee osteoarthritis — Results: "
    )
    assert chunks[0].section_path == "Results"


def test_nested_sections_produce_a_breadcrumb_path() -> None:
    xml = _article(
        f"<sec><title>Results</title>"
        f"<sec><title>Grading</title><p>{LONG_A}</p></sec></sec>"
    )
    chunks = chunk_jats(xml, MANIFEST)
    assert chunks[0].section_path == "Results > Grading"


def test_chunks_never_cross_a_section_boundary() -> None:
    xml = _article(
        f"<sec><title>Methods</title><p>{LONG_A}</p></sec>"
        f"<sec><title>Results</title><p>{LONG_B}</p></sec>"
    )
    chunks = chunk_jats(xml, MANIFEST)
    paths = {c.section_path for c in chunks}
    assert paths == {"Methods", "Results"}
    for chunk in chunks:
        assert chunk.section_path in chunk.text.split(": ", 1)[0]


def test_long_sections_split_into_multiple_chunks() -> None:
    body = f"<sec><title>Discussion</title><p>{LONG_A}</p><p>{LONG_B}</p><p>{LONG_C}</p></sec>"
    chunks = chunk_jats(_article(body), MANIFEST)
    assert len(chunks) >= 2
    assert all(c.section_path == "Discussion" for c in chunks)


def test_chunk_ids_are_deterministic_and_unique() -> None:
    body = f"<sec><title>Discussion</title><p>{LONG_A}</p><p>{LONG_B}</p><p>{LONG_C}</p></sec>"
    first = chunk_jats(_article(body), MANIFEST)
    second = chunk_jats(_article(body), MANIFEST)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(c.chunk_id.startswith("PMC123456#") for c in first)


# ── Inline cleanup: what makes a sentence usable as an NLI premise ────────────


def test_xref_text_is_deleted_not_unwrapped() -> None:
    """A dangling "[12]" or "Fig. 3" poisons a sentence used as a premise."""
    xml = _article(
        f"<sec><title>Results</title>"
        f'<p>Osteophytes were graded <xref ref-type="bibr" rid="b12">[12]</xref> '
        f"independently. {LONG_A}</p></sec>"
    )
    text = chunk_jats(xml, MANIFEST)[0].text
    assert "[12]" not in text
    assert "Osteophytes were graded independently." in text


def test_formatting_elements_are_unwrapped_keeping_text() -> None:
    xml = _article(
        f"<sec><title>Results</title>"
        f"<p>The <italic>medial</italic> compartment showed <bold>marked</bold> "
        f"narrowing. {LONG_A}</p></sec>"
    )
    text = chunk_jats(xml, MANIFEST)[0].text
    assert "medial compartment showed marked narrowing" in text


def test_figure_and_table_captions_are_kept_but_markup_discarded() -> None:
    caption = (
        "Joint space width measurements across Kellgren-Lawrence grades zero "
        "through four, showing the progressive medial compartment narrowing that "
        "characterises advancing radiographic disease in this observational cohort."
    )
    xml = _article(
        f"<sec><title>Results</title><p>{LONG_A}</p>"
        f"<table-wrap><caption><p>{caption}</p></caption>"
        f"<table><tr><td>99</td><td>12</td></tr></table></table-wrap></sec>"
    )
    joined = " ".join(c.text for c in chunk_jats(xml, MANIFEST))
    assert "Joint space width measurements" in joined
    assert "99" not in joined


# ── Exclusions ────────────────────────────────────────────────────────────────


def test_back_matter_is_excluded() -> None:
    """Reference lists match every query lexically and entail nothing."""
    refs = (
        "<ref-list><ref><mixed-citation>Kellgren JH, Lawrence JS. Radiological "
        "assessment of osteo-arthrosis. Ann Rheum Dis. 1957;16(4):494-502. "
        "A reference list entry long enough to survive the length filter if it "
        "were ever considered as a candidate chunk at all.</mixed-citation>"
        "</ref></ref-list>"
    )
    xml = _article(f"<sec><title>Results</title><p>{LONG_A}</p></sec>", back=refs)
    joined = " ".join(c.text for c in chunk_jats(xml, MANIFEST))
    assert "Ann Rheum Dis" not in joined


@pytest.mark.parametrize(
    "title",
    [
        "Acknowledgements",
        "Acknowledgments",
        "Competing interests",
        "Author contributions",
        "Funding",
        "Data availability",
        "Ethics approval and consent to participate",
        "Conflict of interest",
    ],
)
def test_administrative_sections_are_dropped(title: str) -> None:
    assert section_is_dropped(title) is True


@pytest.mark.parametrize(
    "title",
    ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusions"],
)
def test_substantive_sections_are_kept(title: str) -> None:
    """Methods in particular carries MRI protocol detail relevant to SKM-TEA queries."""
    assert section_is_dropped(title) is False


def test_dropped_section_content_does_not_reach_chunks() -> None:
    xml = _article(
        f"<sec><title>Results</title><p>{LONG_A}</p></sec>"
        f"<sec><title>Competing interests</title><p>The authors declare that they "
        f"have no competing interests whatsoever in relation to this submitted "
        f"manuscript or any of the clinical findings reported herein.</p></sec>"
    )
    joined = " ".join(c.text for c in chunk_jats(xml, MANIFEST))
    assert "competing interests" not in joined.lower()


# ── Quality filters ───────────────────────────────────────────────────────────


def test_short_paragraphs_are_filtered_out() -> None:
    xml = _article("<sec><title>Results</title><p>See below.</p></sec>")
    assert chunk_jats(xml, MANIFEST) == []


def test_digit_heavy_debris_is_filtered_out() -> None:
    debris = " ".join(["12.4 0.98 3.1 44.0 7.77 0.012 19.3 88.1 2.04 0.55"] * 12)
    xml = _article(f"<sec><title>Results</title><p>{debris}</p></sec>")
    assert chunk_jats(xml, MANIFEST) == []


def test_chunk_respects_the_hard_character_ceiling() -> None:
    from rag.config import CHUNK_MAX_CHARS

    body = "<sec><title>Discussion</title>" + "".join(
        f"<p>{LONG_A}</p>" for _ in range(8)
    ) + "</sec>"
    for chunk in chunk_jats(_article(body), MANIFEST):
        assert chunk.char_len <= CHUNK_MAX_CHARS


# ── Provenance ────────────────────────────────────────────────────────────────


def test_chunks_inherit_licence_and_provenance_from_the_manifest() -> None:
    xml = _article(f"<sec><title>Results</title><p>{LONG_A}</p></sec>")
    chunk = chunk_jats(xml, MANIFEST)[0]
    assert chunk.licence == "https://creativecommons.org/licenses/by/4.0/"
    assert chunk.source_type == "pmc"
    assert chunk.source_id == "PMC123456"
    assert chunk.url == MANIFEST.source_url


def test_chunking_refuses_an_article_whose_licence_does_not_verify() -> None:
    """The licence gate runs before any parsing, not as a later filter."""
    from rag.licensing import LicenceNotRedistributable

    nc = _article(
        f"<sec><title>Results</title><p>{LONG_A}</p></sec>",
        permissions="""<permissions><license>
          <ali:license_ref>https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref>
        </license></permissions>""",
    )
    with pytest.raises(LicenceNotRedistributable):
        chunk_jats(nc, MANIFEST)
