"""Tests for prompt construction.

Pure function, no model, no network. The load-bearing test here is the ablation
invariant: the grounded and ungrounded prompts must be byte-identical apart from
the evidence block, or any measured difference between the two configurations
cannot be attributed to retrieval.
"""

from __future__ import annotations

import pytest

from rag.prompt import (
    NOTE_JSON_SCHEMA,
    build_messages,
    parse_note,
    render_user_message,
)
from rag.schemas import (
    FeedbackGenerateRequest,
    LabelProbability,
    PatientContext,
    RetrievedPassage,
)

PASSAGES = [
    RetrievedPassage(
        passage_id="PMC1#00-00",
        source="Knee grading — Results",
        text="Definite osteophytes with medial joint space narrowing indicate grade 3.",
        score=0.81,
        licence="https://creativecommons.org/licenses/by/4.0/",
        source_type="pmc",
        url="https://example.org/PMC1",
    ),
    RetrievedPassage(
        passage_id="kl-grading#02-00",
        source="KL grading — Grade 3",
        text="Subchondral sclerosis is detectable in the medial tibial plateau.",
        score=0.77,
        licence="internal",
        source_type="internal",
    ),
]


def mri_request(**kwargs) -> FeedbackGenerateRequest:
    defaults = dict(
        modality="mri",
        grade=3,
        confidence=81.4,
        positive_labels=[
            LabelProbability(name="Effusion", probability=84.7),
            LabelProbability(name="Meniscal Tear (Radial)", probability=91.2),
        ],
        patient_context=PatientContext(age=65, gender="Male", bmi=27.3, pain_level=6),
    )
    defaults.update(kwargs)
    return FeedbackGenerateRequest(**defaults)  # type: ignore[arg-type]


# ── The ablation invariant ────────────────────────────────────────────────────


def test_ungrounded_and_grounded_prompts_differ_only_by_evidence() -> None:
    """Otherwise a measured difference between configs is not attributable
    to retrieval."""
    req = mri_request()
    grounded = render_user_message(req, PASSAGES, include_passages=True)
    ungrounded = render_user_message(req, PASSAGES, include_passages=False)

    assert "## EVIDENCE" in grounded
    assert "## EVIDENCE" not in ungrounded

    def without_evidence(text: str) -> str:
        out, skipping = [], False
        for line in text.splitlines():
            if line.startswith("## EVIDENCE"):
                skipping = True
                continue
            if skipping and line.startswith("## "):
                skipping = False
            if not skipping:
                out.append(line)
        return "\n".join(out)

    assert without_evidence(grounded) == without_evidence(ungrounded)


def test_system_message_is_identical_across_configs() -> None:
    req = mri_request()
    grounded = build_messages(req, PASSAGES, include_passages=True)
    ungrounded = build_messages(req, PASSAGES, include_passages=False)
    assert grounded[0] == ungrounded[0]
    assert grounded[0]["role"] == "system"


# ── Findings block ────────────────────────────────────────────────────────────


def test_labels_are_sorted_by_probability_descending() -> None:
    """The model must hedge differently on a 51% label than a 94% one."""
    text = render_user_message(mri_request(), PASSAGES, include_passages=True)
    assert text.index("Meniscal Tear (Radial)") < text.index("Effusion")
    assert "91.2%" in text and "84.7%" in text


def test_xray_requests_omit_the_label_block_entirely() -> None:
    """Sending "none" would invite comment on absent soft-tissue findings."""
    req = FeedbackGenerateRequest(modality="xray", grade=2, confidence=88.0)
    text = render_user_message(req, PASSAGES, include_passages=True)
    assert "SKM-TEA" not in text
    assert "Kellgren-Lawrence grade: 2" in text


def test_low_confidence_is_flagged_inline() -> None:
    """Max-softmax confidence is poorly calibrated; the prompt says so."""
    low = render_user_message(mri_request(confidence=58.0), PASSAGES, True)
    high = render_user_message(mri_request(confidence=96.0), PASSAGES, True)
    assert "low" in low.lower()
    assert "low" not in high.split("## PATIENT")[0].lower()


# ── Patient block ─────────────────────────────────────────────────────────────


def test_absent_patient_fields_are_omitted_not_rendered_unknown() -> None:
    req = mri_request(patient_context=PatientContext(age=40))
    text = render_user_message(req, PASSAGES, include_passages=True)
    assert "unknown" not in text.lower()
    assert "BMI" not in text
    assert "40" in text


def test_no_patient_context_omits_the_whole_block() -> None:
    req = mri_request(patient_context=None)
    text = render_user_message(req, PASSAGES, include_passages=True)
    assert "## PATIENT" not in text


# ── Determinism ───────────────────────────────────────────────────────────────


def test_rendering_is_deterministic() -> None:
    """Re-runs must be byte-identical for the evaluation to be reproducible."""
    req = mri_request()
    first = render_user_message(req, PASSAGES, include_passages=True)
    second = render_user_message(req, PASSAGES, include_passages=True)
    assert first == second


def test_passages_are_numbered_from_one() -> None:
    text = render_user_message(mri_request(), PASSAGES, include_passages=True)
    assert "[1]" in text and "[2]" in text
    assert "[0]" not in text


# ── Output parsing ────────────────────────────────────────────────────────────


def test_parses_well_formed_json() -> None:
    note = parse_note(
        '{"synthesis": "Grade 3 changes. [1]", "next_steps": ["Refer. [2]"]}'
    )
    assert note is not None
    assert note.synthesis == "Grade 3 changes. [1]"
    assert note.next_steps == ["Refer. [2]"]


def test_parses_json_wrapped_in_prose() -> None:
    raw = 'Here is the note:\n{"synthesis": "A finding.", "next_steps": []}\nDone.'
    note = parse_note(raw)
    assert note is not None
    assert note.synthesis == "A finding."


def test_falls_back_to_treating_plain_prose_as_the_synthesis() -> None:
    """Tolerant rather than failing — but the caller counts these separately."""
    note = parse_note("The knee shows moderate degenerative change.")
    assert note is not None
    assert "moderate degenerative change" in note.synthesis
    assert note.next_steps == []


def test_returns_none_for_empty_output() -> None:
    assert parse_note("") is None
    assert parse_note("   \n  ") is None


def test_truncated_json_degrades_to_prose_rather_than_raising() -> None:
    note = parse_note('{"synthesis": "Grade 3 chan')
    assert note is not None


def test_json_schema_declares_the_expected_fields() -> None:
    props = NOTE_JSON_SCHEMA["properties"]
    assert "synthesis" in props
    assert "next_steps" in props
