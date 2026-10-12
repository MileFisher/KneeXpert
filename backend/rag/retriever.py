"""Turn a diagnostic payload into retrieved passages.

The payload is serialised into one natural-language sentence before embedding.
That same sentence is reused as the reference text for answer-relevance scoring,
so note-versus-query similarity is measured against prose rather than against a
serialised dictionary — the embedder was trained on the former.
"""

from __future__ import annotations

from rag.config import MAX_CHUNKS_PER_ARTICLE, TOP_K
from rag.prompt import GRADE_LABEL
from rag.schemas import FeedbackGenerateRequest
from rag.store import RetrievedPassage, get_store

MODALITY_PHRASE = {"xray": "on radiography", "mri": "on MRI"}


def build_query(req: FeedbackGenerateRequest) -> str:
    """Render the payload as one clinical sentence.

    Deterministic: labels are sorted, so the same payload always produces the
    same query and therefore the same retrieved set.
    """
    grade_label = GRADE_LABEL.get(req.grade, "unknown grade").lower()
    parts = [
        f"Kellgren-Lawrence grade {req.grade} ({grade_label}) knee osteoarthritis "
        f"{MODALITY_PHRASE.get(req.modality, '')}".strip()
    ]

    if req.positive_labels:
        ordered = sorted(
            req.positive_labels, key=lambda label: (-label.probability, label.name)
        )
        names = [label.name for label in ordered]
        parts.append("with " + ", ".join(names))

    ctx = req.patient_context
    if ctx is not None:
        facts = []
        if ctx.age is not None:
            facts.append(f"a {ctx.age}-year-old")
        if ctx.gender:
            facts.append(str(ctx.gender).lower())
        if ctx.bmi is not None:
            facts.append(f"with BMI {ctx.bmi:.1f}")
        if facts:
            parts.append("in " + " ".join(facts))

    return " ".join(parts)


def _cap_per_source(
    passages: list[RetrievedPassage], limit: int
) -> list[RetrievedPassage]:
    """Keep at most ``limit`` passages per source document, preserving rank order.

    Stops one verbose article owning the whole context window, which both wastes
    tokens and narrows the evidence the model can draw on.
    """
    seen: dict[str, int] = {}
    kept: list[RetrievedPassage] = []
    for passage in passages:
        source_id = passage.passage_id.split("#", 1)[0]
        count = seen.get(source_id, 0)
        if count >= limit:
            continue
        seen[source_id] = count + 1
        kept.append(passage)
    return kept


def retrieve(req: FeedbackGenerateRequest, k: int = TOP_K) -> list[RetrievedPassage]:
    """Retrieve passages for a payload.

    Over-fetches before capping per source, so the cap removes duplicates from one
    article rather than shrinking the final set below ``k``.
    """
    from rag.embedder import get_embedder

    store = get_store()
    vector = get_embedder().embed_query(build_query(req))
    hits = store.search(vector, k=k * MAX_CHUNKS_PER_ARTICLE + k)
    return _cap_per_source(hits, MAX_CHUNKS_PER_ARTICLE)[:k]
