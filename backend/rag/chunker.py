"""JATS XML -> retrievable chunks. Pure: no network, no disk, no models.

Research articles are far messier than the clinical guidelines this corpus was
originally meant to hold, so the cleanups here carry most of the retrieval
quality. Each is a deliberate decision:

* **Licence first.** Nothing is parsed from an article whose licence does not
  verify, so a non-redistributable article cannot leak in through this path.
* **Back matter is dropped wholesale.** A reference list matches almost every
  query lexically and entails nothing, making it the single biggest source of
  retrieval garbage in a PMC corpus.
* **Cross-reference text is deleted, not unwrapped.** A sentence left holding a
  dangling "[12]" or "Fig. 3" is poisoned as a natural-language-inference premise.
* **Every chunk carries a context header.** A mid-article paragraph saying "these
  lesions" has no antecedent on its own; naming the article and section does more
  for retrieval quality than tuning the retrieval depth.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from rag.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MAX_DIGIT_RATIO,
    CHUNK_MAX_NON_ALPHA_RATIO,
    CHUNK_MIN_CHARS,
    CHUNK_TARGET_CHARS,
)
from rag.corpus_types import Chunk, ManifestEntry
from rag.licensing import JATS_NS, assert_commercial_licence

# Subtrees that never contain clinically useful prose.
DROPPED_TAGS: frozenset[str] = frozenset(
    {
        "ref-list",
        "fn-group",
        "supplementary-material",
        "disp-formula",
        "inline-formula",
        "table",  # the markup; <caption> is handled separately
        "tex-math",
        "mml:math",
        "graphic",
        "media",
        "funding-group",
    }
)

# Inline wrappers whose text should survive but whose markup should not.
UNWRAPPED_TAGS: frozenset[str] = frozenset(
    {"italic", "bold", "sup", "sub", "sc", "underline", "ext-link", "monospace"}
)

# Administrative sections, matched on the title rather than on sec-type, because
# sec-type is inconsistently populated across publishers.
DROPPED_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^acknowledge?ments?$", re.I),
    re.compile(r"^competing\s+interests?$", re.I),
    re.compile(r"^conflicts?\s+of\s+interest", re.I),
    re.compile(r"^author\s+contributions?$", re.I),
    re.compile(r"^authors?'?\s+contributions?$", re.I),
    re.compile(r"^funding", re.I),
    re.compile(r"^data\s+availability", re.I),
    re.compile(r"^ethics?\b", re.I),
    re.compile(r"^consent\b", re.I),
    re.compile(r"^abbreviations?$", re.I),
    re.compile(r"^supplementary", re.I),
)

_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]")

# Markdown inline markup, stripped from internal documents for the same reason
# <xref> text is dropped from JATS: residual markers degrade a chunk both as a
# retrieval target and as an entailment premise.
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_STRONG = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_MD_EMPHASIS = re.compile(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)", re.S)
_MD_CODE = re.compile(r"`([^`]*)`")


def context_header(title: str, section_path: str) -> str:
    """The prefix every chunk carries, naming its article and section.

    Defined once: the format is also parsed back off a chunk when displaying a
    retrieved passage, so the two must not drift.
    """
    return f"{title} — {section_path}: " if section_path else f"{title}: "


def strip_context_header(text: str) -> str:
    """Recover a chunk's body, tolerating titles that themselves contain ": "."""
    marker = ": "
    idx = text.find(" — ")
    if idx != -1:
        after = text[idx + 3 :]
        pos = after.find(marker)
        if pos != -1:
            return after[pos + len(marker) :]
    pos = text.find(marker)
    return text[pos + len(marker) :] if pos != -1 else text


def section_is_dropped(title: str) -> bool:
    """True when a section title marks administrative rather than clinical content.

    Methods is deliberately *kept*: it carries imaging-protocol detail that is
    directly relevant to MRI queries.
    """
    stripped = title.strip()
    return any(pattern.match(stripped) for pattern in DROPPED_SECTION_PATTERNS)


def strip_markdown_inline(raw: str) -> str:
    """Remove inline markdown markup, keeping the text it wrapped.

    Link text is kept and the URL discarded, since a bare URL inside a sentence
    is noise to both the embedder and the entailment model.
    """
    text = _MD_LINK.sub(r"\1", raw)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_STRONG.sub(r"\2", text)
    return _MD_EMPHASIS.sub(r"\2", text)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalise_text(raw: str) -> str:
    """Collapse whitespace and repair punctuation left behind by removed elements."""
    text = raw.replace("­", "").replace(" ", " ").replace(" ", " ")
    text = _WHITESPACE.sub(" ", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def _element_text(node: ET.Element) -> str:
    """Flatten an element to prose, applying the inline rules.

    ``<xref>`` contributes nothing — not even its tail-adjacent spacing — which is
    what removes citation markers cleanly rather than leaving brackets behind.
    """
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        name = _localname(child.tag)
        if name == "xref":
            pass  # drop the marker entirely
        elif name in DROPPED_TAGS:
            pass
        elif name in UNWRAPPED_TAGS:
            parts.append(_element_text(child))
        else:
            parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _is_quality_chunk(body: str) -> bool:
    """Reject table debris, statistics dumps and fragments too short to stand alone.

    Takes the chunk body *without* its context header. The header is article
    metadata, not content, so including it would let a long title carry a
    too-short passage past the minimum-length filter.
    """
    if len(body) < CHUNK_MIN_CHARS:
        return False
    stripped = body.replace(" ", "")
    if not stripped:
        return False
    non_alpha = sum(1 for ch in stripped if not ch.isalpha())
    if non_alpha / len(stripped) > CHUNK_MAX_NON_ALPHA_RATIO:
        return False
    digits = sum(1 for ch in stripped if ch.isdigit())
    return digits / len(stripped) <= CHUNK_MAX_DIGIT_RATIO


def _paragraphs_of_section(sec: ET.Element) -> list[str]:
    """Direct prose of a section: its own paragraphs and captions, not sub-sections."""
    blocks: list[str] = []
    for child in sec:
        name = _localname(child.tag)
        if name == "p":
            blocks.append(_element_text(child))
        elif name in {"table-wrap", "fig", "boxed-text"}:
            for caption in child.iter():
                if _localname(caption.tag) == "caption":
                    blocks.append(_element_text(caption))
                    break
    return [normalise_text(b) for b in blocks if normalise_text(b)]


def _pack(blocks: list[str]) -> list[str]:
    """Pack paragraphs up to the target size, overlapping by one paragraph.

    Packing stops at the section boundary by construction: this only ever sees one
    section's blocks.

    Consecutive chunks share their boundary paragraph, so a passage whose meaning
    straddles a boundary stays retrievable from either side. The cost is some
    duplicated text in the index, which is harmless both as a retrieval target and
    as an entailment premise.
    """
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(_split_oversized(block))

    packed: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = current + [piece]
        if current and len(" ".join(candidate)) > CHUNK_TARGET_CHARS:
            packed.append(" ".join(current))
            # Carry the last paragraph forward as the overlap.
            current = [current[-1], piece] if len(current) > 1 else [piece]
        else:
            current = candidate
    if current:
        packed.append(" ".join(current))
    return packed


def _split_oversized(block: str) -> list[str]:
    """Split a single over-long paragraph at sentence boundaries."""
    if len(block) <= CHUNK_TARGET_CHARS:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= CHUNK_TARGET_CHARS:
            current = f"{current} {sentence}"
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out


def _walk_sections(
    node: ET.Element, trail: tuple[str, ...], default_title: str = ""
) -> list[tuple[str, list[str]]]:
    """Depth-first walk yielding ``(section_path, paragraphs)`` in document order.

    ``default_title`` names the node when it carries no ``<title>`` of its own —
    an abstract, for instance, which is titled by its element rather than by a
    child.
    """
    collected: list[tuple[str, list[str]]] = []
    own_title = default_title
    for child in node:
        if _localname(child.tag) == "title":
            own_title = normalise_text(_element_text(child))
            break

    if own_title and section_is_dropped(own_title):
        return []

    path = trail + (own_title,) if own_title else trail
    paragraphs = _paragraphs_of_section(node)
    if paragraphs:
        collected.append((" > ".join(p for p in path if p), paragraphs))

    for child in node:
        if _localname(child.tag) == "sec":
            collected.extend(_walk_sections(child, path))
    return collected


def _article_title(root: ET.Element, fallback: str) -> str:
    for node in root.iterfind("./front/article-meta/title-group/article-title"):
        title = normalise_text(_element_text(node))
        if title:
            return title
    return fallback


def chunk_jats(xml_bytes: bytes, manifest: ManifestEntry) -> list[Chunk]:
    """Turn one article's JATS XML into retrievable chunks.

    The licence is asserted before anything is parsed, so this function raises
    rather than returning an empty list for a non-redistributable article.

    Raises:
        LicenceNotRedistributable: if the article's licence does not verify.
    """
    assertion = assert_commercial_licence(
        manifest.pmcid, xml_bytes, oa_bucket=manifest.oa_bucket
    )

    root = ET.fromstring(xml_bytes)
    title = _article_title(root, manifest.title)

    sections: list[tuple[str, list[str]]] = []
    for abstract in root.iterfind("./front/article-meta/abstract"):
        # _walk_sections already emits this node's own paragraphs before
        # recursing, so collecting them separately would duplicate every
        # unstructured abstract.
        sections.extend(_walk_sections(abstract, (), default_title="Abstract"))

    for body in root.iterfind("./body"):
        direct = _paragraphs_of_section(body)
        if direct:
            sections.append(("Body", direct))
        for child in body:
            if _localname(child.tag) == "sec":
                sections.extend(_walk_sections(child, ()))

    chunks: list[Chunk] = []
    for sec_index, (section_path, paragraphs) in enumerate(sections):
        for chunk_index, packed in enumerate(_pack(paragraphs)):
            if not _is_quality_chunk(packed):
                continue
            text = context_header(title, section_path) + packed
            if len(text) > CHUNK_MAX_CHARS:
                text = text[:CHUNK_MAX_CHARS].rsplit(" ", 1)[0]
            chunks.append(
                Chunk(
                    chunk_id=f"{manifest.pmcid}#{sec_index:02d}-{chunk_index:02d}",
                    source_id=manifest.pmcid,
                    title=title,
                    section_path=section_path,
                    text=text,
                    source_type="pmc",
                    licence=assertion.licence_ref,
                    url=manifest.source_url,
                )
            )
    return chunks


def chunk_markdown(path: Path, licence: str = "internal") -> list[Chunk]:
    """Chunk a self-authored internal document by its ``##`` headings.

    Internal documents carry ``source_type="internal"`` so the evaluation can
    report what fraction of retrieved passages were self-authored rather than
    drawn from published literature.
    """
    raw = path.read_text(encoding="utf-8")
    stem = path.stem
    lines = raw.splitlines()

    doc_title = stem.replace("-", " ").replace("_", " ").title()
    for line in lines:
        if line.startswith("# "):
            doc_title = strip_markdown_inline(line[2:].strip())
            break

    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer and not section_is_dropped(current_heading):
            body = normalise_text(strip_markdown_inline(" ".join(buffer)))
            if body:
                sections.append((current_heading, [body]))

    for line in lines:
        if line.startswith("## "):
            flush()
            buffer.clear()
            current_heading = strip_markdown_inline(line[3:].strip())
        elif line.startswith("# "):
            continue
        else:
            buffer.append(line)
    flush()

    chunks: list[Chunk] = []
    for sec_index, (heading, paragraphs) in enumerate(sections):
        for chunk_index, packed in enumerate(_pack(paragraphs)):
            if not _is_quality_chunk(packed):
                continue
            text = context_header(doc_title, heading) + packed
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}#{sec_index:02d}-{chunk_index:02d}",
                    source_id=stem,
                    title=doc_title,
                    section_path=heading,
                    text=text,
                    source_type="internal",
                    licence=licence,
                    url=None,
                )
            )
    return chunks
