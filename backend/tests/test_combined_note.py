"""Tests for the combined X-ray + MRI note: contract, prompt, premise, retrieval.

Pure: no models, no index. The endpoint path is covered in
test_feedback_endpoint.py with the other integration tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.prompt import (
    MRI_GRADE_CAVEAT,
    build_messages,
    render_findings_premise,
    render_user_message,
)
from rag.retriever import build_query, pinned_grades
from rag.schemas import FeedbackGenerateRequest, RetrievedPassage

XRAY = {"modality": "xray", "grade": 2, "confidence": 90.0}
MRI = {
    "modality": "mri",
    "grade": 4,
    "confidence": 60.0,
    "positive_labels": [
        {"name": "Effusion", "probability": 86.0},
        {"name": "Cartilage Lesion (3)", "probability": 94.0},
    ],
}


def combined(**overrides) -> FeedbackGenerateRequest:
    data = {
        "modality": "combined",
        "grade": 0,
        "confidence": 0,
        "studies": [XRAY, MRI],
        **overrides,
    }
    return FeedbackGenerateRequest.model_validate(data)


# ── Contract ──────────────────────────────────────────────────────────────────


def test_the_grade_comes_from_the_orchestrator_not_the_client() -> None:
    req = combined(grade=0, confidence=12.0)
    assert (req.grade, req.confidence) == (3, 75.0)  # (2*90 + 4*60) / 150 = 2.8


def test_labels_come_from_the_mri_study() -> None:
    names = [label.name for label in combined().positive_labels]
    assert sorted(names) == ["Cartilage Lesion (3)", "Effusion"]


@pytest.mark.parametrize(
    "studies",
    [[XRAY], [XRAY, XRAY], [MRI, MRI], []],
    ids=["xray-only", "two-xray", "two-mri", "none"],
)
def test_a_combined_note_needs_one_study_of_each_kind(studies) -> None:
    with pytest.raises(ValidationError):
        combined(studies=studies)


def test_a_single_scan_note_may_not_carry_studies() -> None:
    with pytest.raises(ValidationError):
        FeedbackGenerateRequest(
            modality="xray", grade=2, confidence=90, studies=[XRAY]  # type: ignore[list-item]
        )


def test_single_scan_requests_are_unchanged() -> None:
    req = FeedbackGenerateRequest(modality="xray", grade=2, confidence=90)
    assert req.fusion is None and req.studies == []


# ── Prompt ────────────────────────────────────────────────────────────────────


def test_findings_name_both_studies_the_fused_grade_and_the_gap() -> None:
    text = render_user_message(combined(), [], include_passages=False)
    assert "Kellgren-Lawrence grade 2 (Mild)" in text
    assert "grade 4 (Severe)" in text
    assert "Combined grade (confidence-weighted): 3 (Moderate)" in text
    assert "disagree by 2 grades" in text
    assert MRI_GRADE_CAVEAT in text


def test_agreement_is_stated_when_the_grades_match() -> None:
    text = render_user_message(
        combined(studies=[XRAY, {**MRI, "grade": 2}]), [], include_passages=False
    )
    assert "The X-ray and MRI grades agree." in text


def test_the_low_confidence_study_is_flagged() -> None:
    text = render_user_message(combined(), [], include_passages=False)
    mri_line = next(line for line in text.splitlines() if line.startswith("MRI"))
    assert "materially uncertain" in mri_line


def test_combined_prompts_keep_the_ablation_invariant() -> None:
    """Only the evidence block may differ between grounded and ungrounded."""
    passage = RetrievedPassage(
        passage_id="kl-grading#04-00",
        source="KL",
        text="Definite narrowing.",
        score=1.0,
        licence="internal",
        source_type="internal",
    )
    grounded = build_messages(combined(), [passage], include_passages=True)
    ungrounded = build_messages(combined(), [passage], include_passages=False)
    assert grounded[0] == ungrounded[0]
    without = grounded[1]["content"].replace(
        "## EVIDENCE\n[1] (KL)\n    Definite narrowing.\n\n", ""
    )
    assert without == ungrounded[1]["content"]


# ── Gate premise ──────────────────────────────────────────────────────────────


def test_the_premise_lets_the_gate_verify_statements_about_both_studies() -> None:
    premise = render_findings_premise(combined())
    assert "On X-ray, the Kellgren-Lawrence grade is 2" in premise
    assert "On MRI, the grade is 4" in premise
    assert "disagree by 2 grades" in premise
    assert MRI_GRADE_CAVEAT in premise


# ── Retrieval ─────────────────────────────────────────────────────────────────


def test_the_xray_grade_definition_is_pinned_first() -> None:
    assert pinned_grades(combined()) == [2, 3]  # X-ray, then the fused grade


def test_one_definition_is_pinned_when_xray_and_fused_grades_match() -> None:
    req = combined(studies=[XRAY, {**MRI, "grade": 2}])
    assert pinned_grades(req) == [2]


def test_single_scan_notes_pin_their_own_grade() -> None:
    assert pinned_grades(FeedbackGenerateRequest(modality="mri", grade=3, confidence=80)) == [3]


def test_the_query_names_both_modalities() -> None:
    assert "on radiography and MRI" in build_query(combined())
