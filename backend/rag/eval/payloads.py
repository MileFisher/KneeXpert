"""The evaluation payload set. Pure and deterministic.

    python -m rag.eval.payloads            # write payloads.json
    python -m rag.eval.payloads --summary  # describe the set without writing

Payloads are *constructed* rather than taken from real pipeline output. The
object under evaluation is the clinical-note layer, not the vision models —
those are measured elsewhere — and constructing them means the set covers rare
pathologies evenly instead of inheriting whatever a 36-study test split happens
to contain. It also needs no GPU and no patient data to reproduce.

The enumeration is fixed and ordered, so the committed ``payloads.json`` is a
reviewable artefact and a grader can rebuild it byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, TypedDict

from rag.config import EVAL_PAYLOADS_PATH


class LabelSpec(TypedDict):
    name: str
    probability: float


class PayloadCase(TypedDict):
    """One evaluation case, matching FeedbackGenerateRequest plus a case id."""

    case_id: str
    modality: str
    grade: int
    confidence: float
    positive_labels: list[LabelSpec]
    patient_context: dict[str, Any]

# Confidence bands, chosen to straddle the prompt's own calibration cut-offs:
# below 70 is flagged as materially uncertain, 85 and above as high.
CONFIDENCE_BANDS: tuple[tuple[str, float], ...] = (
    ("low", 61.0),
    ("mid", 79.5),
    ("high", 94.0),
)


@dataclass(frozen=True)
class PatientProfile:
    """A clinical context the note is expected to adapt to."""

    key: str
    age: int
    gender: str
    bmi: float
    pain_level: int
    history: str
    symptoms: str


PROFILES: tuple[PatientProfile, ...] = (
    PatientProfile(
        key="young-athlete",
        age=28,
        gender="Male",
        bmi=22.4,
        pain_level=4,
        history="Competitive football. Acute twisting injury 3 weeks ago.",
        symptoms="Mechanical catching on pivot; no night pain.",
    ),
    PatientProfile(
        key="midlife-raised-bmi",
        age=52,
        gender="Female",
        bmi=31.2,
        pain_level=7,
        history="Insidious onset over 4 years. No prior surgery.",
        symptoms="Pain on stairs and after prolonged standing.",
    ),
    PatientProfile(
        key="older-prior-surgery",
        age=65,
        gender="Male",
        bmi=27.3,
        pain_level=6,
        history="Bilateral knee pain, 3 years. Prior meniscal repair, right knee.",
        symptoms="Morning stiffness under 30 minutes; pain on stairs.",
    ),
    PatientProfile(
        key="elderly-comorbid",
        age=78,
        gender="Female",
        bmi=24.1,
        pain_level=8,
        history="Type 2 diabetes, hypertension. Reduced mobility 18 months.",
        symptoms="Constant rest pain; uses a frame indoors.",
    ),
)

# SKM-TEA label combinations that actually co-occur, keyed by the KL grade they
# plausibly accompany. Names match backend/mri/categories.json exactly.
#
# Probabilities are set so each combination exercises the prompt's requirement to
# hedge differently on a marginal label than on a confident one.
GRADE_LABELS: dict[int, tuple[tuple[tuple[str, float], ...], ...]] = {
    0: (
        (),
        (("Effusion", 62.0),),
    ),
    1: (
        (("Meniscal Tear (Myxoid)", 58.0),),
        (("Cartilage Lesion (1)", 64.0), ("Effusion", 55.0)),
    ),
    2: (
        (("Meniscal Tear (Horizontal)", 77.0), ("Cartilage Lesion (2A)", 69.0)),
        (("Cartilage Lesion (2A)", 81.0), ("Effusion", 72.0)),
        (("Ligament Tear (Low-Grade Sprain)", 66.0),),
        # Vertical/longitudinal: preserves hoop stress and is more often
        # repairable, so it belongs with mild structural change.
        (("Meniscal Tear (Vertical/Longitudinal)", 83.0),),
    ),
    3: (
        (
            ("Meniscal Tear (Radial)", 91.2),
            ("Effusion", 84.7),
            ("Cartilage Lesion (2B)", 66.1),
        ),
        (
            ("Meniscal Tear (Complex)", 88.0),
            ("Cartilage Lesion (2B)", 74.0),
        ),
        (
            ("Ligament Tear (Moderate Grade Sprain or Mucoid Degeneration)", 79.0),
            ("Effusion", 68.0),
        ),
        # Oblique (parrot-beak): shares features of radial and longitudinal
        # patterns, so it is the remaining tear morphology to cover.
        (
            ("Meniscal Tear (Oblique)", 86.0),
            ("Cartilage Lesion (2A)", 71.0),
        ),
    ),
    4: (
        (
            ("Cartilage Lesion (3)", 94.0),
            ("Meniscal Tear (Extrusion)", 89.0),
            ("Effusion", 86.0),
        ),
        (
            ("Meniscal Tear (Flap)", 85.0),
            ("Cartilage Lesion (3)", 90.0),
        ),
        (
            ("Ligament Tear (Full Thickness/Complete Tear)", 92.0),
            ("Cartilage Lesion (3)", 87.0),
        ),
    ),
}


def _case_id(
    modality: str, grade: int, combo_index: int, band: str, profile: str
) -> str:
    """Stable identifier. Used as the resume key, so it must not drift."""
    return f"{modality}-g{grade}-c{combo_index}-{band}-{profile}"


def build_payloads() -> list[PayloadCase]:
    """Enumerate the payload set in a fixed order.

    X-ray payloads carry no labels (the modality cannot see soft tissue), so they
    vary only over grade, confidence and patient profile. MRI payloads add the
    label combinations. Profiles rotate rather than multiply out, which keeps the
    set at a size one GPU session can run three times over without losing
    coverage of any grade or band.
    """
    payloads: list[PayloadCase] = []

    # X-ray: 5 grades x 3 bands x rotating profile = 15
    for grade in range(5):
        for band_index, (band, confidence) in enumerate(CONFIDENCE_BANDS):
            profile = PROFILES[(grade + band_index) % len(PROFILES)]
            payloads.append(
                {
                    "case_id": _case_id("xray", grade, 0, band, profile.key),
                    "modality": "xray",
                    "grade": grade,
                    "confidence": confidence,
                    "positive_labels": [],
                    "patient_context": {
                        "age": profile.age,
                        "gender": profile.gender,
                        "bmi": profile.bmi,
                        "pain_level": profile.pain_level,
                        "history": profile.history,
                        "symptoms": profile.symptoms,
                    },
                }
            )

    # MRI: 5 grades x their combinations x 3 bands x rotating profile
    for grade in range(5):
        for combo_index, combo in enumerate(GRADE_LABELS[grade]):
            for band_index, (band, confidence) in enumerate(CONFIDENCE_BANDS):
                profile = PROFILES[
                    (grade + combo_index + band_index) % len(PROFILES)
                ]
                payloads.append(
                    {
                        "case_id": _case_id(
                            "mri", grade, combo_index, band, profile.key
                        ),
                        "modality": "mri",
                        "grade": grade,
                        "confidence": confidence,
                        "positive_labels": [
                            {"name": name, "probability": probability}
                            for name, probability in combo
                        ],
                        "patient_context": {
                            "age": profile.age,
                            "gender": profile.gender,
                            "bmi": profile.bmi,
                            "pain_level": profile.pain_level,
                            "history": profile.history,
                            "symptoms": profile.symptoms,
                        },
                    }
                )

    return payloads


def write_payloads(path=EVAL_PAYLOADS_PATH) -> int:
    """Write the set as formatted JSON. Returns the number of cases."""
    payloads = build_payloads()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payloads, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(payloads)


def load_payloads(path=EVAL_PAYLOADS_PATH) -> list[PayloadCase]:
    """Read the committed set, falling back to generating it if absent."""
    if not path.is_file():
        return build_payloads()
    return list(json.loads(path.read_text(encoding="utf-8")))


def summarise(payloads: list[PayloadCase]) -> str:
    """Human-readable coverage report, for checking the set before a paid run."""
    by_modality: dict[str, int] = {}
    by_grade: dict[int, int] = {}
    labels_seen: set[str] = set()
    for case in payloads:
        modality = case["modality"]
        by_modality[modality] = by_modality.get(modality, 0) + 1
        by_grade[case["grade"]] = by_grade.get(case["grade"], 0) + 1
        for label in case["positive_labels"]:
            labels_seen.add(label["name"])

    lines = [f"{len(payloads)} cases"]
    lines.append(
        "  by modality: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_modality.items()))
    )
    lines.append(
        "  by grade:    " + ", ".join(f"{k}={v}" for k, v in sorted(by_grade.items()))
    )
    lines.append(f"  distinct SKM-TEA labels covered: {len(labels_seen)}/16")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", action="store_true", help="describe the set without writing it"
    )
    args = parser.parse_args(argv)

    payloads = build_payloads()
    print(summarise(payloads))
    if not args.summary:
        count = write_payloads()
        print(f"\nWrote {count} cases to {EVAL_PAYLOADS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
