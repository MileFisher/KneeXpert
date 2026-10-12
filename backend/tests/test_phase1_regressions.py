"""Regressions for bugs found in the Phase 1 code review."""

from __future__ import annotations

from typing import Any

import pytest

from rag.schemas import FeedbackGenerateRequest, LabelProbability

PAYLOAD = FeedbackGenerateRequest(
    modality="mri",
    grade=3,
    confidence=81.4,
    positive_labels=[LabelProbability(name="Effusion", probability=84.7)],
)


# ── Premise symmetry across configurations ────────────────────────────────────


def test_retry_retrieval_is_not_gated_on_prompt_inclusion(monkeypatch) -> None:
    """Every config must be scored against the same premise set.

    The final attempt re-retrieves more widely. Gating that on whether passages
    go *into the prompt* meant the grounded configs ended up scored against ten
    premises and the ungrounded config against six — so a groundedness difference
    between them would partly measure premise count, not retrieval.
    """
    from rag import service

    ks: list[int] = []

    def _retrieve(req: Any, k: int = 6) -> list[Any]:
        ks.append(k)
        return []

    monkeypatch.setattr("rag.retriever.retrieve", _retrieve)
    monkeypatch.setattr(service, "_to_schema", lambda ps: list(ps))

    grounded = service._retrieval_ks_for_attempt(attempt=2, include_passages=True)
    ungrounded = service._retrieval_ks_for_attempt(attempt=2, include_passages=False)
    assert grounded == ungrounded, "premise breadth must not depend on the config"


# ── Zero-hit retrieval ────────────────────────────────────────────────────────


def test_zero_hits_withholds_the_note_rather_than_demanding_citations(
    monkeypatch,
) -> None:
    """A non-empty index can still return nothing for a query.

    Rendering "(no passages retrieved)" while the system rules still require a
    [n] citation on every claim invites the model to invent markers.
    """
    from rag import ollama_client, service
    from rag.store import get_store

    monkeypatch.setattr(type(get_store()), "count", lambda self: 42)
    monkeypatch.setattr("rag.retriever.retrieve", lambda req, k=6: [])
    monkeypatch.setattr(ollama_client, "is_available", lambda *a, **k: True)

    result = service.generate_clinical_note(PAYLOAD)
    assert result.clinical_note is None
    assert result.fallback_reason == "empty_corpus"


# ── Parse provenance is reported ──────────────────────────────────────────────


def _stub_gate(monkeypatch, score: float = 0.95) -> None:
    """Replace entailment scoring with a fixed verdict.

    These tests are about how output was parsed, not about grounding, and the
    real cross-encoder is ~740 MB and hundreds of milliseconds per pair.
    """
    from rag.gate import SentenceScore

    monkeypatch.setattr(
        "rag.service.groundedness",
        lambda sentences, passages: [
            SentenceScore(
                text=text,
                sentence_class=cls,
                groundedness=score,
                label="entailment",
                supporting_passage_id="x#00-00",
            )
            for text, cls in sentences
        ],
    )
    monkeypatch.setattr("rag.service.answer_relevance", lambda q, a: 0.8)


def test_prose_fallback_is_recorded_not_silently_accepted(monkeypatch) -> None:
    """A rising prose-fallback rate is a signal about the model, not a detail."""
    _stub_gate(monkeypatch)
    from rag import ollama_client, service
    from rag.store import RetrievedPassage, get_store

    passage = RetrievedPassage(
        passage_id="x#00-00",
        source="S",
        text="t",
        score=0.5,
        licence="internal",
        source_type="internal",
    )
    monkeypatch.setattr(type(get_store()), "count", lambda self: 1)
    monkeypatch.setattr("rag.retriever.retrieve", lambda req, k=6: [passage])
    monkeypatch.setattr(ollama_client, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(
        ollama_client, "chat", lambda *a, **k: "Moderate degenerative change."
    )

    result = service.generate_clinical_note(PAYLOAD)
    assert result.clinical_note is not None
    assert result.parse_mode == "prose_fallback"


def test_well_formed_json_is_recorded_as_such(monkeypatch) -> None:
    _stub_gate(monkeypatch)
    from rag import ollama_client, service
    from rag.store import RetrievedPassage, get_store

    passage = RetrievedPassage(
        passage_id="x#00-00",
        source="S",
        text="t",
        score=0.5,
        licence="internal",
        source_type="internal",
    )
    monkeypatch.setattr(type(get_store()), "count", lambda self: 1)
    monkeypatch.setattr("rag.retriever.retrieve", lambda req, k=6: [passage])
    monkeypatch.setattr(ollama_client, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(
        ollama_client,
        "chat",
        lambda *a, **k: '{"synthesis": "A finding [1].", "next_steps": []}',
    )

    result = service.generate_clinical_note(PAYLOAD)
    assert result.parse_mode == "json"


# ── Retry independence ────────────────────────────────────────────────────────


def test_each_attempt_uses_a_distinct_seed(monkeypatch) -> None:
    """A fixed seed with an identical prompt makes a retry near-pointless.

    Attempts must differ by more than temperature, while staying reproducible.
    """
    from rag import ollama_client, service
    from rag.store import get_store

    seeds: list[int] = []

    def _chat(messages, **kwargs):
        seeds.append(kwargs["seed"])
        return ""  # never parses, so every attempt is used

    monkeypatch.setattr(type(get_store()), "count", lambda self: 1)
    monkeypatch.setattr(
        "rag.retriever.retrieve",
        lambda req, k=6: [
            __import__("rag.store", fromlist=["RetrievedPassage"]).RetrievedPassage(
                passage_id="x#00-00", source="S", text="t", score=0.5,
                licence="internal", source_type="internal",
            )
        ],
    )
    monkeypatch.setattr(ollama_client, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(ollama_client, "chat", _chat)

    service.generate_clinical_note(PAYLOAD)
    assert len(seeds) >= 2
    assert len(set(seeds)) == len(seeds), f"seeds repeated across attempts: {seeds}"


# ── Status-code policy ────────────────────────────────────────────────────────


def test_nli_unavailable_is_not_a_runtime_fallback_reason() -> None:
    """Missing entailment weights is a deployment fault, so it must be a 503.

    Listing it as a 200 fallback would contradict the resolved 503-vs-200 rule.
    """
    import typing

    from rag.schemas import FallbackReason

    assert "nli_unavailable" not in typing.get_args(FallbackReason)


# ── No duplicated grade vocabulary ────────────────────────────────────────────


def test_grade_labels_come_from_one_place() -> None:
    """The prompt's grade names were an exact copy of the feedback engine's."""
    import feedback_engine
    from rag.prompt import GRADE_LABEL

    assert GRADE_LABEL is feedback_engine.GRADE_LABELS
