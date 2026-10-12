"""Integration tests for the safety gate against the real entailment model.

These load the real cross-encoder and are therefore slow (a few seconds each).
Marked so they can be excluded during fast iteration:

    python -m pytest -m "not slow"

The unit tests in test_gate.py cover the arithmetic; what these establish is the
thing no synthetic score can: that a fabricated clinical claim actually scores
below a supported one on the real model. If that ordering does not hold, every
threshold above it is theatre.
"""

from __future__ import annotations

import pytest

from rag.config import rag_index_ready
from rag.schemas import RetrievedPassage

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not rag_index_ready(),
        reason="no retrieval index; run `python -m rag.ingest` first",
    ),
]

PASSAGES = [
    RetrievedPassage(
        passage_id="kl-grading#04-00",
        source="KL grading — Grade 3",
        text=(
            "Definite joint space narrowing, usually medial. Osteophyte formation "
            "is present at multiple margins, which may include the tibial plateau, "
            "the femoral condyles and the patella. Subchondral sclerosis is "
            "detectable, characteristically in the medial tibial plateau."
        ),
        score=0.81,
        licence="internal",
        source_type="internal",
    ),
    RetrievedPassage(
        passage_id="kl-grading#07-00",
        source="KL grading — Reliability",
        text=(
            "KL grading is a reading of structure, not of symptoms, and the "
            "correlation between radiographic grade and reported pain is weak. "
            "Inter-observer variance is substantial."
        ),
        score=0.77,
        licence="internal",
        source_type="internal",
    ),
]

SUPPORTED = (
    "Definite joint space narrowing is present in the medial compartment.",
    "synthesis",
)
FABRICATED = (
    "Serum CRP of 48 mg/L confirms septic arthritis requiring intravenous "
    "vancomycin within 72 hours.",
    "synthesis",
)


def test_a_supported_claim_outscores_a_fabricated_one() -> None:
    """The ordering the whole gate rests on."""
    from rag.triad import groundedness

    scores = groundedness([SUPPORTED, FABRICATED], PASSAGES)  # type: ignore[list-item]
    supported, fabricated = scores
    assert supported.groundedness > fabricated.groundedness, (
        f"supported={supported.groundedness:.3f} "
        f"fabricated={fabricated.groundedness:.3f}"
    )


def test_each_sentence_is_attributed_to_a_real_passage() -> None:
    """The surfaced citation comes from entailment, so it cannot be invented."""
    from rag.triad import groundedness

    ids = {p.passage_id for p in PASSAGES}
    for score in groundedness([SUPPORTED], PASSAGES):  # type: ignore[list-item]
        assert score.supporting_passage_id in ids


def test_scoring_with_no_passages_yields_zero_not_an_error() -> None:
    """The ungrounded configuration must still produce scores to report."""
    from rag.triad import groundedness

    scores = groundedness([SUPPORTED], [])  # type: ignore[list-item]
    assert len(scores) == 1
    assert scores[0].groundedness == 0.0
    assert scores[0].supporting_passage_id is None


def test_note_sentences_classifies_synthesis_and_next_steps() -> None:
    from rag.triad import note_sentences

    items = note_sentences(
        "Grade 3 change is present [1]. Sclerosis is detectable [2].",
        ["Refer for orthopaedic review [1]."],
    )
    assert [cls for _, cls in items] == ["synthesis", "synthesis", "next_step"]


def test_a_multi_sentence_next_step_stays_one_unit() -> None:
    """Splitting a recommendation would let half of it pass the gate."""
    from rag.triad import note_sentences

    items = note_sentences("", ["Refer onward. Then review in six weeks."])
    assert len([1 for _, cls in items if cls == "next_step"]) == 1


def test_context_relevance_is_none_without_context() -> None:
    """Undefined, not zero: zero would read as "retrieval failed" in results."""
    from rag.triad import context_relevance

    assert context_relevance([]) is None
    assert context_relevance(PASSAGES) == pytest.approx(0.79)
