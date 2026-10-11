"""Qdrant vector store, embedded on-disk mode.

Real Qdrant, but with no server and no container: the client is given a path and
manages local segments. At a few hundred chunks a network hop buys nothing.

Point ids are a UUID5 of the deterministic ``chunk_id``, so re-ingesting the same
corpus overwrites rather than duplicating.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rag.config import COLLECTION, INDEX_DIR
from rag.corpus_types import Chunk

if TYPE_CHECKING:  # import cost is paid lazily in load()
    from qdrant_client import QdrantClient

_NAMESPACE = uuid.NAMESPACE_URL


def point_id(chunk_id: str) -> str:
    """Stable Qdrant point id derived from a chunk id, making re-ingest idempotent."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


@dataclass(frozen=True)
class RetrievedPassage:
    """One search hit, carrying enough provenance to cite it without a second lookup."""

    passage_id: str
    source: str
    text: str
    score: float
    licence: str
    source_type: str
    url: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "source": self.source,
            "text": self.text,
            "score": round(self.score, 6),
            "licence": self.licence,
            "source_type": self.source_type,
            "url": self.url,
        }


class VectorStore:
    """Embedded Qdrant collection holding the clinical corpus."""

    def __init__(self) -> None:
        self.path = INDEX_DIR
        self.collection = COLLECTION
        self.client: QdrantClient | None = None
        self._loaded = False

    def load(self) -> None:
        """Open the on-disk store. Idempotent."""
        if self._loaded:
            return
        from qdrant_client import QdrantClient

        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self._loaded = True

    def close(self) -> None:
        """Release the on-disk lock. Qdrant's local mode is single-process."""
        if self.client is not None:
            self.client.close()
        self.client = None
        self._loaded = False

    def recreate(self, dim: int) -> None:
        """Drop and recreate the collection for a clean rebuild."""
        self.load()
        from qdrant_client.models import Distance, VectorParams

        assert self.client is not None
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def ensure(self, dim: int) -> None:
        """Create the collection only if it does not already exist."""
        self.load()
        assert self.client is not None
        if not self.client.collection_exists(self.collection):
            self.recreate(dim)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Insert or replace chunks. Returns the number of points written."""
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        self.load()
        from qdrant_client.models import PointStruct

        assert self.client is not None
        points = [
            PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "section_path": chunk.section_path,
                    "text": chunk.text,
                    "source_type": chunk.source_type,
                    "licence": chunk.licence,
                    "url": chunk.url,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def count(self) -> int:
        """Number of indexed points, or 0 when the collection does not exist."""
        self.load()
        assert self.client is not None
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedPassage]:
        """Nearest passages to a query vector, best first."""
        self.load()
        assert self.client is not None
        if not self.client.collection_exists(self.collection):
            return []
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
        ).points

        passages: list[RetrievedPassage] = []
        for hit in hits:
            payload = hit.payload or {}
            title = payload.get("title", "")
            section = payload.get("section_path", "")
            source = f"{title} — {section}" if section else str(title)
            passages.append(
                RetrievedPassage(
                    passage_id=str(payload.get("chunk_id", hit.id)),
                    source=source,
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                    licence=str(payload.get("licence", "")),
                    source_type=str(payload.get("source_type", "")),
                    url=payload.get("url"),
                )
            )
        return passages

    def source_type_breakdown(self) -> dict[str, int]:
        """Point count per source type, so the evaluation can report the corpus mix."""
        self.load()
        assert self.client is not None
        if not self.client.collection_exists(self.collection):
            return {}
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        counts: dict[str, int] = {}
        for source_type in ("pmc", "internal"):
            result = self.client.count(
                self.collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_type", match=MatchValue(value=source_type)
                        )
                    ]
                ),
                exact=True,
            )
            if result.count:
                counts[source_type] = int(result.count)
        return counts


_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Module-level cached store."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
