"""Corpus vocabulary — the records that flow from harvest through ingest.

One module so that ``harvest``, ``chunker`` and ``ingest`` share a single
definition of what an article and a chunk are, rather than passing dicts around.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SourceType = Literal["pmc", "internal"]


@dataclass(frozen=True)
class ManifestEntry:
    """Provenance for one corpus article.

    Written to ``corpus/manifest.json`` and committed, so that the licence of
    every passage the system can cite is auditable without re-fetching anything.
    """

    pmcid: str
    title: str
    licence_ref: str
    licence_source: str
    """Which JATS construct carried the licence: "ali:license_ref" or "xlink:href"."""
    sha256: str
    source_url: str
    retrieved_at: str
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    oa_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestEntry:
        fields = {k: raw.get(k) for k in cls.__dataclass_fields__}
        return cls(**fields)  # type: ignore[arg-type]

    @property
    def citation(self) -> str:
        """Short human-readable source label, shown beside a retrieved passage."""
        parts = [self.title]
        if self.journal:
            parts.append(self.journal)
        if self.year:
            parts.append(str(self.year))
        return ", ".join(parts)


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage.

    ``text`` already carries its context header (``"{title} — {section_path}: "``),
    because a mid-article paragraph is frequently unusable both as a retrieval
    target and as an entailment premise without it.
    """

    chunk_id: str
    source_id: str
    """PMC id for literature, filename stem for internal documents."""
    title: str
    section_path: str
    text: str
    source_type: SourceType
    licence: str
    url: str | None = None

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["char_len"] = self.char_len
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Chunk:
        fields = {
            k: raw.get(k) for k in cls.__dataclass_fields__ if k in raw or k != "url"
        }
        fields.pop("char_len", None)
        return cls(**fields)  # type: ignore[arg-type]


def write_chunks(chunks: list[Chunk], out_path: Path) -> None:
    """Write chunks as JSONL sorted by ``chunk_id``.

    Sorting makes the committed file diff cleanly, so a reviewer can see exactly
    what changed in the corpus without rebuilding the index.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(chunks, key=lambda c: c.chunk_id)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in ordered:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def read_chunks(path: Path) -> list[Chunk]:
    """Read a chunks JSONL file written by :func:`write_chunks`."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
