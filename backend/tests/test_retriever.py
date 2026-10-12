"""Tests for query building and grade-definition pinning."""

from __future__ import annotations

import pytest

from rag.config import rag_index_ready
from rag.retriever import build_query, grade_section_for
from rag.schemas import FeedbackGenerateRequest, LabelProbability, PatientContext


# ── Query construction ────────────────────────────────────────────────────────


def test_query_is_one_clinical_sentence() -> None:
    q = build_query(
        FeedbackGenerateRequest(
            modality="mri",
            grade=3,
            confidence=80.0,
            positive_labels=[LabelProbability(name="Effusion", probability=84.0)],
            patient_context=PatientContext(age=65, gender="Male", bmi=27.3),
        )
    )
    assert "grade 3" in q
    assert "Effusion" in q
    assert "65-year-old" in q
    assert "\n" not in q


def test_query_is_deterministic_regardless_of_label_order() -> None:
    """Same payload must always retrieve the same set."""
    labels = [
        LabelProbability(name="Effusion", probability=84.0),
        LabelProbability(name="Meniscal Tear (Radial)", probability=91.0),
    ]
    a = build_query(
        FeedbackGenerateRequest(
            modality="mri", grade=3, confidence=80.0, positive_labels=labels
        )
    )
    b = build_query(
        FeedbackGenerateRequest(
            modality="mri",
            grade=3,
            confidence=80.0,
            positive_labels=list(reversed(labels)),
        )
    )
    assert a == b


# ── Grade definitions are looked up, not searched for ─────────────────────────


@pytest.mark.parametrize("grade", [0, 1, 2, 3, 4])
def test_every_grade_maps_to_a_section_name(grade: int) -> None:
    assert grade_section_for(grade).startswith(f"Grade {grade}")


@pytest.mark.skipif(
    not rag_index_ready(), reason="no retrieval index; run `python -m rag.ingest`"
)
def test_the_payload_grade_definition_is_always_retrieved() -> None:
    """Dense search cannot separate the grade sections, so it must not have to.

    Measured: against a grade-3 query the Grade 3 chunk scored 0.795 while Grade 0
    scored 0.764 — a 0.031 spread across eight sections that share their
    vocabulary and a context header worth a fifth of each chunk. The grade is a
    structured field in the payload, so the matching definition is pinned by
    lookup rather than left to similarity.
    """
    from rag.retriever import retrieve

    for grade in (0, 2, 4):
        req = FeedbackGenerateRequest(modality="xray", grade=grade, confidence=80.0)
        ids = [p.passage_id for p in retrieve(req)]
        assert any(
            pid.startswith("kl-grading#") for pid in ids
        ), f"grade {grade}: no KL definition retrieved ({ids})"


@pytest.mark.skipif(
    not rag_index_ready(), reason="no retrieval index; run `python -m rag.ingest`"
)
def test_pinning_does_not_exceed_the_requested_passage_count() -> None:
    from rag.config import TOP_K
    from rag.retriever import retrieve

    req = FeedbackGenerateRequest(modality="xray", grade=3, confidence=80.0)
    assert len(retrieve(req, k=TOP_K)) <= TOP_K


@pytest.mark.skipif(
    not rag_index_ready(), reason="no retrieval index; run `python -m rag.ingest`"
)
def test_pinned_definition_is_not_duplicated_when_search_also_finds_it() -> None:
    from rag.retriever import retrieve

    req = FeedbackGenerateRequest(modality="xray", grade=3, confidence=80.0)
    ids = [p.passage_id for p in retrieve(req)]
    assert len(ids) == len(set(ids))
