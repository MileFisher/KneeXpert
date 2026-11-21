"""A grounded synthesis keeps its note when only some next steps are unsupported.

Regression: the pruning path referenced a threshold constant that had been
renamed, so every note that failed the gate on its first attempt raised a
NameError and the endpoint answered 500 instead of serving or withholding.
"""

from __future__ import annotations

from rag.schemas import FeedbackGenerateRequest

PAYLOAD = FeedbackGenerateRequest(modality="xray", grade=2, confidence=82.0)

NOTE_JSON = (
    '{"synthesis": "Joint space narrowing is consistent with KL grade 2 [1].",'
    ' "next_steps": ["Refer for supervised physiotherapy [1].",'
    ' "Schedule total knee replacement [1]."]}'
)


def _score(text: str) -> float:
    return 0.05 if "replacement" in text else 0.95


def test_unsupported_next_step_is_pruned_not_crashed(monkeypatch) -> None:
    from rag import ollama_client, service
    from rag.gate import SentenceScore
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
    monkeypatch.setattr(ollama_client, "chat", lambda *a, **k: NOTE_JSON)
    monkeypatch.setattr(
        "rag.service.groundedness",
        lambda sentences, passages: [
            SentenceScore(
                text=text,
                sentence_class=cls,
                groundedness=_score(text),
                label="entailment",
                supporting_passage_id="x#00-00",
            )
            for text, cls in sentences
        ],
    )
    monkeypatch.setattr("rag.service.answer_relevance", lambda q, a: 0.8)

    result = service.generate_clinical_note(PAYLOAD)

    assert result.clinical_note is not None
    steps = result.clinical_note.next_steps
    assert len(steps) == 1
    assert "physiotherapy" in steps[0]
    assert result.synthesis_groundedness is not None
    assert result.next_steps_groundedness is not None
