"""Request and response shapes for the clinical-note endpoint.

These are the cross-process contract. The frontend mirror lands in Phase 3;
until then the endpoint declares ``response_model`` so the shape is published in
OpenAPI and Phase 3 has something to generate from. Once the mirror exists in
``frontend/src/lib/diagnosticApi.ts``, changing one means changing the other.

The response is deliberately shaped so that a *withheld* note is still a
well-formed 200: ``clinical_note`` goes null and ``fallback_reason`` says why,
with the retrieved passages still attached. There is no "static note" variant —
the deterministic ``FeedbackBundle`` is always present separately, so substituting
dictionary text for model output would be a lie about provenance rather than a
graceful degradation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NoteConfig = Literal["ungrounded", "rag_no_gate", "rag_gate"]
"""Which pipeline to run. Only the evaluation varies this; production uses rag_gate."""

FallbackReason = Literal[
    "llm_unavailable",
    "timeout",
    "gate_rejected",
    "parse_failed",
    "empty_corpus",
]
"""Runtime conditions only. Deployment faults an operator must fix -- a missing
index, missing entailment weights -- are 503s, not fallback reasons."""

ParseMode = Literal["json", "prose_fallback"]
"""How the model's output was read. A rising prose_fallback rate says something
about the model and must not vanish into a successful response."""


class PatientContext(BaseModel):
    """Optional clinical context. Absent fields are omitted from the prompt rather
    than rendered as "unknown", which invites the model to speculate."""

    age: int | None = Field(default=None, ge=0, le=130)
    gender: str | None = None
    bmi: float | None = Field(default=None, ge=0, le=100)
    pain_level: int | None = Field(default=None, ge=0, le=10)
    history: str | None = None
    symptoms: str | None = None


class LabelProbability(BaseModel):
    """One predicted SKM-TEA category. ``probability`` is a percentage, matching
    the scale the MRI pipeline already reports."""

    name: str
    probability: float = Field(ge=0, le=100)


class FeedbackGenerateRequest(BaseModel):
    modality: Literal["xray", "mri"]
    grade: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0, le=100)
    positive_labels: list[LabelProbability] = Field(default_factory=list)
    patient_context: PatientContext | None = None
    config: NoteConfig = "rag_gate"
    case_id: str | None = None
    """Evaluation bookkeeping. Ignored in production."""


class RetrievedPassage(BaseModel):
    passage_id: str
    source: str
    text: str
    score: float
    licence: str
    source_type: str
    url: str | None = None


class NoteSentence(BaseModel):
    """One sentence of the note with its entailment verdict.

    ``supporting_passage_id`` is derived from the entailment model, not from the
    marker the LLM wrote, so it cannot be fabricated. ``cited_marker`` keeps the
    model's own claim for comparison.
    """

    text: str
    groundedness: float
    label: Literal["entailment", "neutral", "contradiction"]
    supporting_passage_id: str | None = None
    cited_marker: int | None = None


class TriadScores(BaseModel):
    """RAG Triad. ``context_relevance`` is None — not zero — for the ungrounded
    configuration, where no context was supplied and the metric is undefined."""

    context_relevance: float | None
    answer_relevance: float
    groundedness: float


class ClinicalNote(BaseModel):
    synthesis: str
    next_steps: list[str] = Field(default_factory=list)
    sentences: list[NoteSentence] = Field(default_factory=list)


class FeedbackGenerateResponse(BaseModel):
    clinical_note: ClinicalNote | None = None
    generated_by: str | None = None
    grounding_score: float | None = None
    """Pooled mean over every sentence. Reported, but no longer what gates."""
    synthesis_groundedness: float | None = None
    next_steps_groundedness: float | None = None
    """None when the note made no recommendations."""
    triad: TriadScores | None = None
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    gate_attempts: int = 0
    gate_threshold: float
    """The synthesis threshold."""
    next_steps_threshold: float | None = None
    fallback_reason: FallbackReason | None = None
    parse_mode: ParseMode | None = None
    latency_ms: dict[str, int] = Field(default_factory=dict)
    config: NoteConfig = "rag_gate"
