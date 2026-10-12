"""Tests for sentence splitting.

The splitter feeds the safety gate: each sentence becomes one entailment
hypothesis. A bad split either fuses two claims (so one unsupported claim hides
behind a supported one) or severs a claim from its own citation, so the edge
cases here are the gate's accuracy, not cosmetics.
"""

from __future__ import annotations

import pytest

from rag.sentences import split_sentences


def test_splits_on_sentence_terminators() -> None:
    text = "Grade 3 change is present. Osteophytes are visible. Refer onward."
    assert split_sentences(text) == [
        "Grade 3 change is present.",
        "Osteophytes are visible.",
        "Refer onward.",
    ]


def test_returns_empty_for_blank_input() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_single_sentence_without_terminator_is_kept() -> None:
    assert split_sentences("Moderate degenerative change") == [
        "Moderate degenerative change"
    ]


# ── Abbreviations: a full stop that is not a sentence end ─────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Findings per Smith et al. are consistent with grade 3.",
        "Imaging modalities, e.g. MRI, show soft-tissue detail.",
        "The medial compartment, i.e. the weight-bearing side, is narrowed.",
        "Osteophytes are visible in Fig. 2 of the report.",
        "Compare radiography vs. magnetic resonance imaging.",
        "Reviewed by Dr. Nguyen before sign-off.",
        "The defect measures approx. 8 mm across.",
        "Measured in mm. across the joint line.",
    ],
)
def test_abbreviations_do_not_end_a_sentence(text: str) -> None:
    assert len(split_sentences(text)) == 1, split_sentences(text)


def test_decimals_do_not_end_a_sentence() -> None:
    text = "Confidence was 81.4% and the BMI was 27.3 at presentation."
    assert len(split_sentences(text)) == 1


def test_abbreviation_mid_text_still_splits_the_real_boundary() -> None:
    text = "Per Smith et al. the grade is 3. Refer for review."
    assert split_sentences(text) == [
        "Per Smith et al. the grade is 3.",
        "Refer for review.",
    ]


# ── Citation markers must stay attached ───────────────────────────────────────


def test_citation_marker_stays_with_its_sentence() -> None:
    """Severing a marker from its claim would misattribute the citation."""
    text = "Joint space narrowing is present [1]. Sclerosis is detectable [2]."
    assert split_sentences(text) == [
        "Joint space narrowing is present [1].",
        "Sclerosis is detectable [2].",
    ]


def test_marker_after_the_full_stop_is_not_orphaned() -> None:
    text = "Grade 3 change is present. [1] Refer for review. [2]"
    parts = split_sentences(text)
    assert all(part.strip() not in {"[1]", "[2]"} for part in parts), parts


# ── Other terminators and whitespace ──────────────────────────────────────────


def test_question_and_exclamation_terminate() -> None:
    assert len(split_sentences("Is this grade 3? It appears so!")) == 2


def test_newlines_are_normalised_not_treated_as_boundaries() -> None:
    text = "Grade 3 change\nis present. Refer\nonward."
    assert split_sentences(text) == ["Grade 3 change is present.", "Refer onward."]


def test_repeated_terminators_do_not_produce_empty_sentences() -> None:
    parts = split_sentences("Grade 3... Refer onward.")
    assert all(part.strip() for part in parts)


def test_splitting_is_deterministic() -> None:
    text = "Grade 3 per Smith et al. is present [1]. Refer onward [2]."
    assert split_sentences(text) == split_sentences(text)
