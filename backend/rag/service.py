"""Orchestrates one clinical-note request. The only module ``app.py`` imports.

Degradation policy, which is the substance of this module:

* **Retrieval always runs**, in every configuration. The ungrounded configuration
  does not put passages in the prompt, but it still needs them as scoring
  premises — otherwise "groundedness" has nothing to score against and the
  ablation is ill-posed.
* **A withheld note is null, never substituted.** There is no static fallback
  text. The deterministic ``FeedbackBundle`` is served separately and always, so
  dressing dictionary output as model output would misreport provenance. The
  response says ``clinical_note: null`` plus a reason, and the UI renders that.
* **Runtime failures are 200s.** Only deployment faults (no index) are errors the
  caller should treat as broken, and those are raised for ``app.py`` to turn into
  a 503.
"""

from __future__ import annotations

import time
from typing import Any

from rag.config import (
    GEN_SEED,
    GROUNDEDNESS_TAU,
    MAX_RETRIES,
    OLLAMA_MODEL,
    RETRY_TEMPERATURES,
    TOP_K,
    TOP_K_RETRY,
)
from rag.prompt import NOTE_JSON_SCHEMA, build_messages, parse_note
from rag.schemas import (
    FeedbackGenerateRequest,
    FeedbackGenerateResponse,
    RetrievedPassage as PassageSchema,
)
from rag.store import RetrievedPassage


class CorpusNotIndexed(RuntimeError):
    """No retrieval index is present. A deployment fault, not a runtime condition."""


def _to_schema(passages: list[RetrievedPassage]) -> list[PassageSchema]:
    """Convert store hits to the response shape.

    Fields are listed explicitly rather than splatted from a dict: the two types
    are field-identical, which is exactly why they were easy to pass
    interchangeably, and spelling them out means a typechecker catches the next
    divergence instead of a cast hiding it.
    """
    return [
        PassageSchema(
            passage_id=p.passage_id,
            source=p.source,
            text=p.text,
            score=p.score,
            licence=p.licence,
            source_type=p.source_type,
            url=p.url,
        )
        for p in passages
    ]


def _retrieval_ks_for_attempt(attempt: int, include_passages: bool) -> int:
    """Premise breadth for one attempt.

    Deliberately ignores ``include_passages``. Whether passages go *into the
    prompt* is what distinguishes the configurations; how many are retrieved as
    scoring premises must not, or a groundedness difference between configs would
    partly measure premise count rather than retrieval.
    """
    return TOP_K_RETRY if attempt >= MAX_RETRIES else TOP_K


def generate_clinical_note(
    req: FeedbackGenerateRequest,
) -> FeedbackGenerateResponse:
    """Retrieve, generate, and (from Phase 2) gate one clinical note.

    Raises:
        CorpusNotIndexed: the vector index is missing or empty.
    """
    from rag import ollama_client
    from rag.retriever import retrieve
    from rag.store import get_store

    latency: dict[str, int] = {}
    started = time.monotonic()

    def mark(stage: str, since: float) -> None:
        latency[stage] = int((time.monotonic() - since) * 1000)

    # ── Retrieval: always, regardless of configuration ────────────────────────
    t = time.monotonic()
    if get_store().count() == 0:
        raise CorpusNotIndexed(
            "Retrieval index is empty. Run `python -m rag.ingest` in backend/."
        )
    # Converted to the response shape immediately, so exactly one passage type
    # flows through the rest of this function. The store's dataclass and the
    # pydantic schema have the same fields, which made them easy to mix up — and
    # duck typing hid it until a typechecker ran.
    passages = _to_schema(retrieve(req, k=TOP_K))
    mark("retrieval", t)

    def respond(**kwargs: Any) -> FeedbackGenerateResponse:
        mark("total", started)
        kwargs.setdefault("retrieved_passages", passages)
        return FeedbackGenerateResponse(
            gate_threshold=GROUNDEDNESS_TAU,
            config=req.config,
            latency_ms=latency,
            **kwargs,
        )

    # A populated index can still return nothing for a given query. Rendering
    # "(no passages retrieved)" while the system rules demand a citation on every
    # claim is an invitation to invent markers, so withhold instead.
    if not passages:
        return respond(fallback_reason="empty_corpus")

    if not ollama_client.is_available():
        return respond(fallback_reason="llm_unavailable")

    include_passages = req.config != "ungrounded"
    attempts = 0
    t = time.monotonic()

    for attempt in range(MAX_RETRIES + 1):
        attempts = attempt + 1
        temperature = RETRY_TEMPERATURES[min(attempt, len(RETRY_TEMPERATURES) - 1)]

        # The final attempt re-retrieves more widely: if the note could not be
        # grounded in the original passages, offering the same ones again is the
        # least likely thing to help. Applied in every configuration -- see
        # _retrieval_ks_for_attempt.
        wanted_k = _retrieval_ks_for_attempt(attempt, include_passages)
        if wanted_k != TOP_K:
            passages = _to_schema(retrieve(req, k=wanted_k))

        messages = build_messages(req, passages, include_passages)
        try:
            raw = ollama_client.chat(
                messages,
                temperature=temperature,
                # A fixed seed with an unchanged prompt makes a retry nearly
                # pointless. Offsetting by the attempt keeps retries independent
                # while staying reproducible for a given payload.
                seed=GEN_SEED + attempt,
                json_schema=NOTE_JSON_SCHEMA,
            )
        except ollama_client.OllamaTimeout:
            mark("generation", t)
            return respond(fallback_reason="timeout", gate_attempts=attempts)
        except ollama_client.OllamaUnavailable:
            mark("generation", t)
            return respond(fallback_reason="llm_unavailable", gate_attempts=attempts)

        note, parse_mode = parse_note(raw)
        if note is None:
            continue

        mark("generation", t)
        # Phase 2 inserts the safety gate here. Until then a parsed note is
        # returned ungated, and `grounding_score` stays None rather than being
        # filled with a placeholder that would read as a real measurement.
        return respond(
            clinical_note=note,
            generated_by=f"{OLLAMA_MODEL} via Ollama",
            gate_attempts=attempts,
            parse_mode=parse_mode,
        )

    mark("generation", t)
    return respond(fallback_reason="parse_failed", gate_attempts=attempts)
