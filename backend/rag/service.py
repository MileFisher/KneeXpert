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
from rag.gate import GateOutcome, SentenceScore, evaluate, plan_attempt
from rag.prompt import (
    NOTE_JSON_SCHEMA,
    build_messages,
    parse_note,
    revision_hint_for,
)
from rag.triad import (
    answer_relevance,
    context_relevance,
    findings_premise,
    groundedness,
    note_sentences,
)
from rag.schemas import (
    ClinicalNote,
    FeedbackGenerateRequest,
    FeedbackGenerateResponse,
    NoteSentence,
    RetrievedPassage as PassageSchema,
    TriadScores,
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


def _with_sentence_scores(
    note: ClinicalNote, scores: list[SentenceScore]
) -> ClinicalNote:
    """Attach per-sentence verdicts to the note the clinician receives.

    The surfaced citation is the entailment-derived passage, not the marker the
    model wrote. ``cited_marker`` keeps the model's own claim alongside it so the
    two can be compared, but only the derived one is authoritative: it cannot be
    fabricated.
    """
    return ClinicalNote(
        synthesis=note.synthesis,
        next_steps=note.next_steps,
        sentences=[
            NoteSentence(
                text=s.text,
                groundedness=round(s.groundedness, 4),
                label=s.label,
                supporting_passage_id=s.supporting_passage_id,
                cited_marker=s.cited_marker,
            )
            for s in scores
        ],
    )


def _triad_for(
    req: FeedbackGenerateRequest,
    note: ClinicalNote,
    passages: list[PassageSchema],
    outcome: GateOutcome,
    include_passages: bool,
) -> TriadScores:
    """All three RAG Triad dimensions. Only groundedness gated the note.

    ``context_relevance`` is None for the ungrounded configuration: no context
    was supplied, so the metric is undefined rather than zero.
    """
    from rag.retriever import build_query

    answer_text = " ".join([note.synthesis, *note.next_steps]).strip()
    return TriadScores(
        context_relevance=context_relevance(passages) if include_passages else None,
        answer_relevance=round(answer_relevance(build_query(req), answer_text), 4),
        groundedness=round(outcome.groundedness, 4),
    )


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
    gate_enabled = req.config == "rag_gate"
    attempts = 0
    generation_ms = 0
    gate_ms = 0
    last_outcome: GateOutcome | None = None
    revision: str | None = None

    for attempt in range(MAX_RETRIES + 1):
        attempts = attempt + 1
        plan = plan_attempt(attempt, MAX_RETRIES)

        # Re-retrieval on the final attempt happens in every configuration, so
        # the premise set is never a function of which config is running — see
        # _retrieval_ks_for_attempt.
        if plan.retrieval_k != TOP_K:
            passages = _to_schema(retrieve(req, k=plan.retrieval_k))

        messages = build_messages(
            req,
            passages,
            include_passages,
            revision_hint=revision if plan.include_revision_hint else None,
        )

        t = time.monotonic()
        try:
            raw = ollama_client.chat(
                messages,
                temperature=plan.temperature,
                # A fixed seed with an unchanged prompt makes a retry nearly
                # pointless. Offsetting by the attempt keeps retries independent
                # while staying reproducible for a given payload.
                seed=GEN_SEED + attempt,
                json_schema=NOTE_JSON_SCHEMA,
            )
        except ollama_client.OllamaTimeout:
            generation_ms += int((time.monotonic() - t) * 1000)
            latency["generation"] = generation_ms
            return respond(fallback_reason="timeout", gate_attempts=attempts)
        except ollama_client.OllamaUnavailable:
            generation_ms += int((time.monotonic() - t) * 1000)
            latency["generation"] = generation_ms
            return respond(fallback_reason="llm_unavailable", gate_attempts=attempts)
        generation_ms += int((time.monotonic() - t) * 1000)

        note, parse_mode = parse_note(raw)
        if note is None:
            continue

        # ── Safety gate ──────────────────────────────────────────────────────
        t = time.monotonic()
        sentences = note_sentences(note.synthesis, note.next_steps)
        # The patient's own findings join the retrieved passages as premises,
        # per §2.9's "patient record and retrieved guidelines". Gate-only:
        # never reported as retrieved, never counted in context relevance.
        premises = [findings_premise(req), *passages]
        scores = groundedness(sentences, premises)
        outcome = evaluate(scores, threshold=GROUNDEDNESS_TAU)
        gate_ms += int((time.monotonic() - t) * 1000)
        last_outcome = outcome

        latency["generation"] = generation_ms
        latency["gate"] = gate_ms

        if outcome.passed or not gate_enabled:
            return respond(
                clinical_note=_with_sentence_scores(note, scores),
                generated_by=f"{OLLAMA_MODEL} via Ollama",
                gate_attempts=attempts,
                parse_mode=parse_mode,
                grounding_score=outcome.groundedness,
                triad=_triad_for(req, note, passages, outcome, include_passages),
            )

        # Name the failing sentences verbatim for the next attempt: a quoted
        # sentence gives the model more to act on than a paraphrased complaint.
        revision = revision_hint_for(outcome.rejected_sentences)

    latency["generation"] = generation_ms
    if last_outcome is None:
        return respond(fallback_reason="parse_failed", gate_attempts=attempts)

    latency["gate"] = gate_ms
    return respond(
        fallback_reason="gate_rejected",
        gate_attempts=attempts,
        grounding_score=last_outcome.groundedness,
    )
