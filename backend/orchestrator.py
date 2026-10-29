"""Central routing module: combine per-modality results into one grade.

The X-ray ensemble and the MRI pipeline each return a Kellgren-Lawrence grade. This
module is the single place they are combined — the frontend used to average them
itself and add a hard-coded confidence bonus, so the rule lived in two copies in
the UI and in none on the server.

Two caveats travel with every fused grade and must be shown wherever it is:

* **The MRI grade is not a KL reading.** ``mri.grade_mapping.multilabel_to_kl``
  infers it from SKM-TEA cartilage-lesion labels (a 2B lesion becomes "KL 3",
  any meniscal tear with no lesion becomes "KL 2"). KL is a radiographic scale.
* **The weights are uncalibrated.** Both confidences are maximum-softmax outputs,
  which the project's own write-up concedes correlate poorly with accuracy.

Confidence weighting was chosen with those caveats known. The fused confidence is
therefore the plain mean of the inputs — agreement between two models is not
evidence that either is more accurate, so nothing is added for it.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Modality = Literal["xray", "mri"]
Agreement = Literal["concordant", "discordant"]


class LabelProbability(BaseModel):
    """One predicted SKM-TEA category. ``probability`` is a percentage, matching
    the scale the MRI pipeline already reports."""

    name: str
    probability: float = Field(ge=0, le=100)


class StudyResult(BaseModel):
    """One modality's output, as the frontend received it from that pipeline."""

    modality: Modality
    grade: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0, le=100)
    positive_labels: list[LabelProbability] = Field(default_factory=list)


class FusionRequest(BaseModel):
    studies: list[StudyResult] = Field(min_length=1, max_length=2)

    @field_validator("studies")
    @classmethod
    def _one_study_per_modality(cls, studies: list[StudyResult]) -> list[StudyResult]:
        modalities = [study.modality for study in studies]
        if len(set(modalities)) != len(modalities):
            raise ValueError("At most one study per modality.")
        return studies


class FusionResult(BaseModel):
    grade: int
    """The confidence-weighted mean, rounded half up."""
    weighted_mean: float
    """Before rounding, so a 2.5 is visibly a coin toss rather than a firm 3."""
    confidence: float
    """Plain mean of the inputs. Never raised for agreement."""
    agreement: Agreement
    grade_gap: int
    weights: dict[str, float]
    """Each modality's share of the weighted mean, summing to 1."""
    studies: list[StudyResult]
    method: Literal["confidence_weighted"] = "confidence_weighted"


def _round_half_up(value: float) -> int:
    """Round .5 upwards.

    Python's ``round`` uses banker's rounding, so ``round(2.5) == 2`` — a
    weighted mean exactly between two grades would resolve towards whichever is
    even. Rounded to 6 places first, because a weighted mean that is 2.5 in exact
    arithmetic can arrive as 2.4999999999999996.
    """
    return math.floor(round(value, 6) + 0.5)


def fuse(studies: list[StudyResult]) -> FusionResult:
    """Combine one or two studies into a single grade.

    A lone study passes through unchanged. With both, each grade is weighted by
    its model's confidence; if every confidence is zero the weights fall back to
    equal rather than dividing by zero.
    """
    if not studies:
        raise ValueError("Nothing to fuse.")

    total = sum(study.confidence for study in studies)
    if total > 0:
        weights = [study.confidence / total for study in studies]
    else:
        weights = [1 / len(studies)] * len(studies)

    weighted_mean = sum(w * study.grade for w, study in zip(weights, studies))
    grades = [study.grade for study in studies]

    return FusionResult(
        grade=_round_half_up(weighted_mean),
        weighted_mean=round(weighted_mean, 4),
        confidence=round(total / len(studies), 1),
        agreement="concordant" if len(set(grades)) == 1 else "discordant",
        grade_gap=max(grades) - min(grades),
        weights={
            study.modality: round(w, 4) for w, study in zip(weights, studies)
        },
        studies=list(studies),
    )
