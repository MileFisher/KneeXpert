"""Prompt construction for the clinical note. Pure: no network, no models.

Meditron3 is continued-pretrained from Llama-3.1-8B-Instruct, so it takes a
two-message chat and responds well to explicitly delimited blocks and negative
constraints.

Two properties matter more than wording:

* **Determinism.** Blocks are rendered in a fixed order from sorted inputs, so
  the same payload produces byte-identical text. Without this the evaluation is
  not reproducible.
* **The ablation invariant.** The evidence block is the *only* thing that differs
  between the grounded and ungrounded configurations. Anything else that varied
  would make a measured difference unattributable to retrieval.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag.schemas import ClinicalNote, FeedbackGenerateRequest, RetrievedPassage

SYSTEM_MESSAGE = """\
You are a musculoskeletal radiology assistant drafting a clinical note for a \
physician to review. You do not make final diagnoses.

RULES
1. Every clinical claim you write must be supported either by the FINDINGS block \
or by a numbered passage in the EVIDENCE block. Nothing else.
2. Cite the supporting passage inline as [n]. A sentence with no support gets no \
citation - and should not be written.
3. Never invent measurements, dosages, laterality, compartment locations, or \
timeframes that are not in FINDINGS or EVIDENCE.
4. Confidence values are maximum-softmax outputs and are poorly calibrated. \
Treat confidence below 70% as materially uncertain and say so.
5. If the evidence does not support a recommendation, write "Insufficient \
evidence in the supplied sources" rather than guessing.
6. Do not restate the patient's identity. Do not address the patient.
7. Output valid JSON only, matching the schema. No prose outside the JSON.

LENGTH: synthesis at most 6 sentences; at most 4 next_steps."""

NOTE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "synthesis": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["synthesis", "next_steps"],
}

MODALITY_LABEL = {
    "xray": "X-ray (CNN ensemble, soft-voting)",
    "mri": "MRI (MACS-Net artifact removal -> DeiT-Small multi-label)",
}

GRADE_LABEL = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}


def _confidence_hint(confidence: float) -> str:
    """Translate a confidence number into a calibration caveat.

    §V of the project's own write-up concedes max-softmax confidence correlates
    poorly with accuracy; this is where that concession becomes model behaviour
    rather than a footnote.
    """
    if confidence < 70:
        return " (low - treat as materially uncertain)"
    if confidence < 85:
        return " (moderate)"
    return " (high)"


def _findings_block(req: FeedbackGenerateRequest) -> list[str]:
    lines = [
        "## FINDINGS",
        f"Modality: {MODALITY_LABEL[req.modality]}",
        f"Kellgren-Lawrence grade: {req.grade} "
        f"({GRADE_LABEL.get(req.grade, 'Unknown')})",
        f"Model confidence: {req.confidence:.1f}%{_confidence_hint(req.confidence)}",
    ]
    # X-ray carries no soft-tissue labels at all. Rendering "none" would invite
    # commentary on findings the modality cannot see.
    if req.modality == "mri" and req.positive_labels:
        lines.append("Positive SKM-TEA categories (probability, descending):")
        ordered = sorted(
            req.positive_labels, key=lambda label: (-label.probability, label.name)
        )
        lines.extend(f"  - {lb.name}: {lb.probability:.1f}%" for lb in ordered)
    return lines


def _patient_block(req: FeedbackGenerateRequest) -> list[str]:
    ctx = req.patient_context
    if ctx is None:
        return []

    facts: list[str] = []
    if ctx.age is not None:
        facts.append(f"Age {ctx.age}")
    if ctx.gender:
        facts.append(str(ctx.gender))
    if ctx.bmi is not None:
        facts.append(f"BMI {ctx.bmi:.1f}")
    if ctx.pain_level is not None:
        facts.append(f"pain {ctx.pain_level}/10")

    lines: list[str] = []
    if facts:
        lines.append(" | ".join(facts))
    if ctx.history:
        lines.append(f"History: {ctx.history}")
    if ctx.symptoms:
        lines.append(f"Symptoms: {ctx.symptoms}")

    # An empty context object must not emit a bare heading.
    return ["## PATIENT", *lines] if lines else []


def _evidence_block(passages: list[RetrievedPassage]) -> list[str]:
    if not passages:
        return ["## EVIDENCE", "(no passages retrieved)"]
    lines = ["## EVIDENCE"]
    for index, passage in enumerate(passages, start=1):
        lines.append(f"[{index}] ({passage.source})")
        lines.append(f"    {passage.text}")
    return lines


# One TASK block for every configuration. An earlier version varied it (dropping
# the citation syntax when no evidence was supplied) and broke the ablation
# invariant: two things differed between configs instead of one, so a measured
# difference could no longer be attributed to retrieval.
#
# Asking the ungrounded configuration for citations it cannot supply is coherent,
# because system rule 2 already says a sentence with no support gets no citation.
# A rule-following model with no evidence block therefore emits no markers.
TASK_BLOCK = """\
## TASK
Write a clinical note as JSON:
{"synthesis": "<paragraph, each sentence ending in its [n] citation>",
 "next_steps": ["<action> [n]", ...]}"""


def render_user_message(
    req: FeedbackGenerateRequest,
    passages: list[RetrievedPassage],
    include_passages: bool,
    revision_hint: str | None = None,
) -> str:
    """Render the user turn.

    ``include_passages`` is the single switch between the grounded and ungrounded
    configurations; everything else is computed identically either way.
    """
    sections: list[list[str]] = [_findings_block(req), _patient_block(req)]
    if include_passages:
        sections.append(_evidence_block(passages))
    sections.append(TASK_BLOCK.splitlines())
    if revision_hint:
        sections.append(["## REVISION", revision_hint])

    return "\n\n".join("\n".join(block) for block in sections if block)


def build_messages(
    req: FeedbackGenerateRequest,
    passages: list[RetrievedPassage],
    include_passages: bool,
    revision_hint: str | None = None,
) -> list[dict[str, str]]:
    """The full chat payload: a fixed system turn plus the rendered user turn."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": render_user_message(
                req, passages, include_passages, revision_hint
            ),
        },
    ]


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
CITATION_MARKER = re.compile(r"\[(\d+)\]")


def parse_note(raw: str) -> ClinicalNote | None:
    """Parse model output into a note, tolerantly.

    Returns None only for genuinely empty output. Anything else yields a note,
    falling back to treating the whole response as the synthesis — but the caller
    records *how* it parsed, because a rising prose-fallback rate is a reportable
    signal about the model rather than a detail to swallow.
    """
    if not raw or not raw.strip():
        return None

    match = _JSON_OBJECT.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            synthesis = payload.get("synthesis")
            steps = payload.get("next_steps") or []
            if isinstance(synthesis, str) and synthesis.strip():
                return ClinicalNote(
                    synthesis=synthesis.strip(),
                    next_steps=[str(s).strip() for s in steps if str(s).strip()],
                )

    return ClinicalNote(synthesis=raw.strip(), next_steps=[])


def revision_hint_for(rejected: list[str]) -> str:
    """The REVISION block used on a retry, naming the unsupported sentences.

    Quoting them verbatim is deliberate: paraphrasing the rejection gives the
    model less to act on than the sentence it actually has to fix.
    """
    quoted = "\n".join(f'  - "{sentence}"' for sentence in rejected)
    return (
        "These statements were not supported by the numbered evidence:\n"
        f"{quoted}\n"
        "Rewrite the note without them, or restate them only in terms the "
        "passages directly support."
    )
