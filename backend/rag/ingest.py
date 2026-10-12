"""Build the retrieval index from the committed corpus.

    python -m rag.ingest              # chunk + embed + index
    python -m rag.ingest --rebuild    # drop the collection first
    python -m rag.ingest --chunks-only  # regenerate chunks.jsonl, no embedding

Chunking is separated from embedding so that ``chunks.jsonl`` can be regenerated
and reviewed in a diff without loading a model, and so that a grader can rebuild
the index from committed artefacts in one command.
"""

from __future__ import annotations

import argparse
import json
import sys

from rag.config import (
    ARTICLES_DIR,
    CHUNKS_PATH,
    INTERNAL_DIR,
    MANIFEST_PATH,
)
from rag.chunker import chunk_jats, chunk_markdown
from rag.corpus_types import Chunk, ManifestEntry, read_chunks, write_chunks
from rag.licensing import LicenceNotRedistributable


def load_manifest() -> list[ManifestEntry]:
    """Read the committed manifest, or return empty when no articles are harvested."""
    if not MANIFEST_PATH.is_file():
        return []
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [ManifestEntry.from_dict(item) for item in raw]


def build_chunks() -> tuple[list[Chunk], list[str]]:
    """Chunk every corpus document. Returns chunks and any problems encountered."""
    chunks: list[Chunk] = []
    problems: list[str] = []

    for entry in load_manifest():
        article_path = ARTICLES_DIR / f"{entry.pmcid}.xml"
        if not article_path.is_file():
            problems.append(
                f"{entry.pmcid}: in manifest but {article_path.name} is missing. "
                "Re-run `python -m rag.harvest`."
            )
            continue
        try:
            chunks.extend(chunk_jats(article_path.read_bytes(), entry))
        except LicenceNotRedistributable as e:
            problems.append(f"{entry.pmcid}: licence no longer verifies: {e}")

    if INTERNAL_DIR.is_dir():
        for path in sorted(INTERNAL_DIR.glob("*.md")):
            chunks.extend(chunk_markdown(path))

    return chunks, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true", help="drop the collection before indexing"
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="write chunks.jsonl without embedding or indexing",
    )
    parser.add_argument(
        "--from-chunks",
        action="store_true",
        help="index the committed chunks.jsonl without re-chunking",
    )
    args = parser.parse_args(argv)

    if args.from_chunks:
        if not CHUNKS_PATH.is_file():
            print(f"No chunk file at {CHUNKS_PATH}. Run without --from-chunks first.")
            return 1
        problems: list[str] = []
        chunks = read_chunks(CHUNKS_PATH)
        print(f"Loaded {len(chunks)} chunk(s) from {CHUNKS_PATH.name}")
    else:
        chunks, problems = build_chunks()
        write_chunks(chunks, CHUNKS_PATH)
        print(f"Wrote {len(chunks)} chunk(s) to {CHUNKS_PATH}")

    by_type: dict[str, int] = {}
    by_licence: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.source_type] = by_type.get(chunk.source_type, 0) + 1
        by_licence[chunk.licence] = by_licence.get(chunk.licence, 0) + 1

    if by_type:
        print("\nChunks by source type:")
        for name, count in sorted(by_type.items()):
            print(f"  {count:4d}  {name}")
        print("Chunks by licence:")
        for name, count in sorted(by_licence.items()):
            print(f"  {count:4d}  {name}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for message in problems:
            print(f"  - {message}")

    if not chunks:
        print("\nNothing to index.")
        return 1 if problems else 0

    if args.chunks_only:
        return 1 if problems else 0

    from rag.embedder import get_embedder
    from rag.store import get_store

    embedder = get_embedder()
    store = get_store()
    try:
        dim = embedder.dim
        if args.rebuild:
            store.recreate(dim)
            print(f"\nRecreated collection (dim={dim})")
        else:
            store.ensure(dim)

        print(f"Embedding {len(chunks)} chunk(s)...")
        vectors = embedder.embed_passages([c.text for c in chunks])
        written = store.upsert(chunks, vectors)
        total = store.count()
        print(f"Indexed {written} point(s); collection now holds {total}")
        breakdown = store.source_type_breakdown()
        if breakdown:
            print("Indexed by source type:")
            for name, count in sorted(breakdown.items()):
                print(f"  {count:4d}  {name}")
    finally:
        store.close()

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
