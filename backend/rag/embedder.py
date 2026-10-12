"""Sentence embeddings for retrieval — local, in-process.

Embedding runs here rather than on the remote generation host so that the index
can be rebuilt offline and retrieval keeps working when that host is unreachable.
Only generation is remote.

Follows the lazy-singleton shape used by ``xray/ensemble.py::get_loader`` and
``mri/deit_classifier.py::get_deit``: construction is cheap, weights load on first
real use, and the instance is cached at module level.
"""

from __future__ import annotations

import numpy as np
import torch

from rag.config import EMBED_BATCH_SIZE, EMBED_MODEL, EMBED_QUERY_PREFIX


class Embedder:
    """BGE-small sentence embedder.

    BGE models are **asymmetric**: a query must carry an instruction prefix while
    a passage must not. The prefix is applied inside :meth:`embed_query` rather
    than by callers, because it is needed at three separate call sites (ingestion,
    retrieval, and answer-relevance scoring) and forgetting it at any one of them
    degrades retrieval silently.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = EMBED_MODEL
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load the model. Idempotent; safe to call before every use."""
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer

        print(f"[rag] loading embedder {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        self._loaded = True

    @property
    def dim(self) -> int:
        """Embedding dimensionality, loading the model if necessary."""
        self.load()
        assert self.model is not None
        # Renamed in sentence-transformers 5.x; the old name still works but warns.
        getter = getattr(
            self.model,
            "get_embedding_dimension",
            getattr(self.model, "get_sentence_embedding_dimension", None),
        )
        if getter is None:
            raise RuntimeError(
                f"{self.model_name}: cannot determine embedding dimensionality"
            )
        return int(getter())

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed corpus passages. No instruction prefix — see the class docstring."""
        self.load()
        assert self.model is not None
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query, applying the required BGE instruction prefix."""
        self.load()
        assert self.model is not None
        vector = self.model.encode(
            EMBED_QUERY_PREFIX + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32)

    def cosine(self, a: str, b: str) -> float:
        """Cosine similarity between two plain texts, both embedded as passages.

        Used for answer relevance and for pre-filtering entailment premises, where
        both sides are prose rather than a query.
        """
        vectors = self.embed_passages([a, b])
        return float(np.dot(vectors[0], vectors[1]))


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Module-level cached embedder."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
