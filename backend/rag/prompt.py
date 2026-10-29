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

from feedback_engine import GRADE_LABELS
from rag.schemas import (
    ClinicalNote,
    FeedbackGenerateRequest,
    ParseMode,
    RetrievedPassage,
)

SYSTEM_MESSAGE = """\
You are a musculoskeletal radiology assistant drafting a clinical note for a \
physician to review. You do not make final diagnoses.

RULES
1. Every clinical claim you write must be supported either by the FINDINGS block \
or by a numbered passage in the EVIDENCE block. Nothing else.
2. Cite the supporting passage inline as [n] using ONLY valid numbers present in \
the EVIDENCE block (e.g. [1] to [6]). Never cite numbers that do not exist in EVIDENCE. \
At most 1-2 citations per sentence. NEVER write repeating sequential citations (e.g. do not write [1][2][3][4][5][6]).
3. Never invent measurements, dosages, laterality, compartment locations, or \
timeframes that are not in FINDINGS or EVIDENCE.
4. Confidence values are maximum-softmax outputs and are poorly calibrated. \
Treat confidence below 70% as materially uncertain and say so.
5. If the evidence does not support a recommendation, write "Insufficient \
evidence in the supplied sources" rather than guessing.
6. Do not restate the patient's identity. Do not address the patient.
7. Output valid JSON only, matching the schema. No prose outside the JSON.

LENGTH: synthesis at most 4 concise sentences; at most 3 next_steps."""

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
    "combined": "X-ray and MRI of the same knee",
}

# Stated as a fact in both the prompt and the gate's findings premise, so the
# model is told it and the gate can verify a sentence that repeats it.
MRI_GRADE_CAVEAT = (
    "The MRI grade is inferred from cartilage-lesion labels; it is not a "
    "Kellgren-Lawrence reading."
)

# Reused rather than restated: the deterministic feedback engine already owns
# the KL grade vocabulary, and two copies would drift.
GRADE_LABEL = GRADE_LABELS


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


def sorted_labels(req: FeedbackGenerateRequest) -> list[Any]:
    """Predicted labels, most probable first, ties broken by name.

    Shared with the retriever so the prompt and the query always agree on order,
    which is part of what makes a run reproducible.
    """
    return sorted(
        req.positive_labels, key=lambda label: (-label.probability, label.name)
    )


def _combined_findings_block(req: FeedbackGenerateRequest) -> list[str]:
    """Both studies, the fused grade, and whether they agree.

    The disagreement is stated outright rather than left for the model to
    notice, because a combined note that quietly reconciles two grades is the
    failure this note exists to avoid.
    """
    xray, mri, fusion = req.study("xray"), req.study("mri"), req.fusion
    assert xray is not None and mri is not None and fusion is not None
    lines = [
        "## FINDINGS",
        f"Modality: {MODALITY_LABEL['combined']}",
        f"{MODALITY_LABEL['xray']}: Kellgren-Lawrence grade {xray.grade} "
        f"({GRADE_LABEL.get(xray.grade, 'Unknown')}), "
        f"confidence {xray.confidence:.1f}%{_confidence_hint(xray.confidence)}",
        f"{MODALITY_LABEL['mri']}: grade {mri.grade} "
        f"({GRADE_LABEL.get(mri.grade, 'Unknown')}), "
        f"confidence {mri.confidence:.1f}%{_confidence_hint(mri.confidence)}",
    ]
    if req.positive_labels:
        lines.append("Positive SKM-TEA categories on MRI (probability, descending):")
        lines.extend(
            f"  - {lb.name}: {lb.probability:.1f}%" for lb in sorted_labels(req)
        )
    lines.append(
        f"Combined grade (confidence-weighted): {fusion.grade} "
        f"({GRADE_LABEL.get(fusion.grade, 'Unknown')})"
    )
    lines.append(_agreement_sentence(fusion.grade_gap))
    lines.append(MRI_GRADE_CAVEAT)
    return lines


def _agreement_sentence(gap: int) -> str:
    if gap == 0:
        return "The X-ray and MRI grades agree."
    return (
        f"The X-ray and MRI grades disagree by {gap} "
        f"grade{'s' if gap != 1 else ''}."
    )


def _findings_block(req: FeedbackGenerateRequest) -> list[str]:
    if req.modality == "combined":
        return _combined_findings_block(req)
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
        ordered = sorted_labels(req)
        lines.extend(f"  - {lb.name}: {lb.probability:.1f}%" for lb in ordered)
    return lines


def render_findings_premise(req: FeedbackGenerateRequest) -> str:
    """The findings and patient blocks as flowing prose.

    Reused as an entailment premise so the gate can verify statements about this
    patient, not only statements about the literature. Rendered as sentences
    rather than the prompt's bullet layout, because an entailment model reads
    prose far better than a list.
    """
    if req.modality == "combined":
        parts = _combined_premise_parts(req)
    else:
        parts = [
            f"This study is a {MODALITY_LABEL[req.modality].split(' (')[0]} "
            "of the knee.",
            f"The Kellgren-Lawrence grade is {req.grade} "
            f"({GRADE_LABEL.get(req.grade, 'unknown')}).",
            f"Model confidence is {req.confidence:.1f} percent"
            f"{_confidence_hint(req.confidence)}.",
        ]
        if req.modality == "mri" and req.positive_labels:
            parts.append(_positive_findings_sentence(req))

    ctx = req.patient_context
    if ctx is not None:
        facts = []
        if ctx.age is not None:
            facts.append(f"aged {ctx.age}")
        if ctx.gender:
            facts.append(str(ctx.gender).lower())
        if ctx.bmi is not None:
            facts.append(f"with a BMI of {ctx.bmi:.1f}")
        if ctx.pain_level is not None:
            facts.append(f"reporting pain {ctx.pain_level} out of 10")
        if facts:
            parts.append("The patient is " + " ".join(facts) + ".")
        if ctx.history:
            parts.append(f"History: {ctx.history}")
        if ctx.symptoms:
            parts.append(f"Symptoms: {ctx.symptoms}")

    return " ".join(parts)


def _positive_findings_sentence(req: FeedbackGenerateRequest) -> str:
    named = ", ".join(
        f"{label.name} at {label.probability:.1f} percent probability"
        for label in sorted_labels(req)
    )
    return f"The following findings are positive: {named}."


def _combined_premise_parts(req: FeedbackGenerateRequest) -> list[str]:
    xray, mri, fusion = req.study("xray"), req.study("mri"), req.fusion
    assert xray is not None and mri is not None and fusion is not None
    parts = [
        "This is a combined X-ray and MRI assessment of the knee.",
        f"On X-ray, the Kellgren-Lawrence grade is {xray.grade} "
        f"({GRADE_LABEL.get(xray.grade, 'unknown')}), with model confidence "
        f"{xray.confidence:.1f} percent{_confidence_hint(xray.confidence)}.",
        f"On MRI, the grade is {mri.grade} "
        f"({GRADE_LABEL.get(mri.grade, 'unknown')}), with model confidence "
        f"{mri.confidence:.1f} percent{_confidence_hint(mri.confidence)}.",
    ]
    if req.positive_labels:
        parts.append(_positive_findings_sentence(req))
    parts.append(
        f"The combined grade, weighted by model confidence, is {fusion.grade} "
        f"({GRADE_LABEL.get(fusion.grade, 'unknown')})."
    )
    parts.append(_agreement_sentence(fusion.grade_gap))
    parts.append(MRI_GRADE_CAVEAT)
    return parts


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


def _clean_citations(text: str, max_passage: int = 10) -> str:
    """Filter out hallucinated citations and collapse runaway repetitive citation loops."""
    if not text:
        return text

    # Remove citations with high numbers (e.g. > max_passage)
    def _filter_num(m: re.Match[str]) -> str:
        num = int(m.group(1))
        return m.group(0) if 1 <= num <= max_passage else ""

    cleaned = CITATION_MARKER.sub(_filter_num, text)

    # Collapse excessive consecutive citations e.g. [1] [2] [3] [4] [5] -> at most 2
    def _collapse_run(m: re.Match[str]) -> str:
        found = CITATION_MARKER.findall(m.group(0))
        unique: list[str] = []
        for c in found:
            marker = f"[{c}]"
            if marker not in unique:
                unique.append(marker)
        return " " + " ".join(unique[:2]) if unique else ""

    cleaned = re.sub(r"(?:\[\d+\]\s*){3,}", _collapse_run, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_note(raw: str) -> tuple[ClinicalNote | None, ParseMode | None]:
    """Parse model output into a note, tolerantly.

    Returns ``(None, None)`` only for genuinely empty output. Anything else
    yields a note paired with the mode it was read in, falling back to treating
    the extracted prose as the synthesis. The mode is returned rather than
    swallowed because a rising prose-fallback rate is a reportable signal about
    the model, and the caller records it on the response.
    """
    if not raw or not raw.strip():
        return None, None

    cleaned_raw = raw.strip()

    match = _JSON_OBJECT.search(cleaned_raw)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            synthesis = payload.get("synthesis")
            steps = payload.get("next_steps") or []
            if isinstance(synthesis, str) and synthesis.strip():
                return (
                    ClinicalNote(
                        synthesis=_clean_citations(synthesis.strip()),
                        next_steps=[
                            _clean_citations(str(s).strip())
                            for s in steps
                            if str(s).strip()
                        ],
                    ),
                    "json",
                )

    # Resilient extraction for malformed or truncated JSON
    if '"synthesis"' in cleaned_raw:
        synth_match = re.search(r'"synthesis"\s*:\s*"((?:\\.|[^"\\])*)', cleaned_raw)
        if synth_match:
            synth_text = synth_match.group(1)
            # Unescape basic JSON string escapes if present
            try:
                synth_text = synth_text.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
            synth_text = _clean_citations(synth_text.strip())

            # Attempt to extract next_steps if any were generated before truncation
            steps: list[str] = []
            steps_match = re.search(r'"next_steps"\s*:\s*\[(.*)', cleaned_raw, re.S)
            if steps_match:
                extracted_steps = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)(?:"|$)', steps_match.group(1))
                steps = [_clean_citations(s.strip()) for s in extracted_steps if s.strip()]

            if synth_text:
                return ClinicalNote(synthesis=synth_text, next_steps=steps), "prose_fallback"

    # Fallback to plain prose: strip any stray JSON wrappers so raw syntax is never shown
    prose = cleaned_raw
    prose = re.sub(r'^\s*\{\s*"synthesis"\s*:\s*"?', '', prose)
    prose = re.sub(r'"?\s*,?\s*"next_steps".*$', '', prose, flags=re.S)
    prose = re.sub(r'[\{\}\[\]"]', '', prose)
    prose = _clean_citations(prose.strip())

    return ClinicalNote(synthesis=prose if prose else cleaned_raw, next_steps=[]), "prose_fallback"


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
