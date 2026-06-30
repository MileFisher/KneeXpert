"""
Unified clinical feedback engine for KneeXpert.

Generates structured, evidence-backed feedback for both X-ray (KL grade)
and MRI (SKM-TEA multi-label → KL mapping) predictions.

Output schema — FeedbackBundle:
{
  "summary": str,              # One-line clinical summary
  "key_findings": [str],       # Detailed findings
  "recommendations": [str],    # Treatment / follow-up recommendations
  "limitations": [str],        # Caveats and uncertainty notes
  "evidence": [str],           # Radiological evidence descriptors
  "sources": [str],            # Reference citations
  "grade_label": str,          # Human-readable grade name
  "severity": str,             # "normal" | "doubtful" | "mild" | "moderate" | "severe"
}
"""

from __future__ import annotations

from typing import Any

# ── KL grade metadata ──────────────────────────────────────────────────────────

GRADE_LABELS: dict[int, str] = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}

SEVERITY_TAGS: dict[int, str] = {
    0: "normal",
    1: "doubtful",
    2: "mild",
    3: "moderate",
    4: "severe",
}

# ── X-ray findings (KL-specific) ───────────────────────────────────────────────

XRAY_FINDINGS: dict[int, list[str]] = {
    0: [
        "No radiographic features of osteoarthritis.",
        "Joint space preserved with no osteophyte formation.",
        "No subchondral sclerosis or bone contour abnormality.",
    ],
    1: [
        "Doubtful joint space narrowing.",
        "Possible early osteophytic lipping at the tibial spines.",
        "No definite subchondral changes.",
    ],
    2: [
        "Definite osteophytes identified at the tibial plateau and femoral condyles.",
        "Possible joint space narrowing in the medial compartment.",
        "Mild subchondral bone changes without sclerosis.",
    ],
    3: [
        "Definite joint space narrowing in the medial compartment.",
        "Osteophyte formation at multiple margins (tibial plateau, femoral condyles, patella).",
        "Subchondral sclerosis detected in the medial tibial plateau.",
        "Possible bone contour deformity.",
    ],
    4: [
        "Marked joint space narrowing with bone-on-bone contact medially.",
        "Large osteophytes at all compartment margins.",
        "Subchondral sclerosis and cyst formation.",
        "Definite bone contour deformity (flattening of femoral condyle).",
    ],
}

# ── MRI pathology-specific findings (SKM-TEA labels) ───────────────────────────

MRI_MENISCAL_FINDINGS: dict[str, list[str]] = {
    "Meniscal Tear (Myxoid)": ["Myxoid degeneration signal within the meniscal substance."],
    "Meniscal Tear (Horizontal)": ["Horizontal cleavage tear pattern identified in the meniscus."],
    "Meniscal Tear (Radial)": ["Radial tear extending to the meniscal capsular junction."],
    "Meniscal Tear (Vertical/Longitudinal)": [
        "Vertical/longitudinal tear along the meniscal circumferential fibers."
    ],
    "Meniscal Tear (Oblique)": ["Oblique tear pattern (parrot-beak configuration)."],
    "Meniscal Tear (Complex)": [
        "Complex tear with multiple planes — increased risk of mechanical symptoms."
    ],
    "Meniscal Tear (Flap)": ["Flap tear with displaced fragment — may cause locking."],
    "Meniscal Tear (Extrusion)": [
        "Meniscal extrusion beyond the tibial plateau margin — indicates root or radial deficiency."
    ],
}

MRI_LIGAMENT_FINDINGS: dict[str, list[str]] = {
    "Ligament Tear (Low-Grade Sprain)": [
        "Low-grade ligament sprain — fibers intact with mild signal abnormality."
    ],
    "Ligament Tear (Moderate Grade Sprain or Mucoid Degeneration)": [
        "Moderate-grade ligament sprain or mucoid degeneration — partial fiber disruption."
    ],
    "Ligament Tear (Full Thickness/Complete Tear)": [
        "Full-thickness ligament tear — complete discontinuity of fibers."
    ],
}

MRI_CARTILAGE_FINDINGS: dict[str, list[str]] = {
    "Cartilage Lesion (1)": [
        "Grade 1 cartilage signal change (softening/fibrillation) — surface intact."
    ],
    "Cartilage Lesion (2A)": [
        "Grade 2A cartilage lesion — superficial partial-thickness defect extending into the superficial zone."
    ],
    "Cartilage Lesion (2B)": [
        "Grade 2B cartilage lesion — deep partial-thickness defect extending into the deep zone."
    ],
    "Cartilage Lesion (3)": [
        "Grade 3 cartilage lesion — full-thickness defect with subchondral bone exposure."
    ],
}

MRI_EFFUSION_FINDINGS: dict[str, list[str]] = {
    "Effusion": [
        "Joint effusion detected — increased intra-articular fluid signal."
    ],
}

# ── Recommendations ────────────────────────────────────────────────────────────

XRAY_RECOMMENDATIONS: dict[int, list[str]] = {
    0: [
        "No specific treatment indicated for osteoarthritis.",
        "Continue general joint health: regular low-impact exercise, weight management.",
        "Re-image only if new symptoms develop.",
    ],
    1: [
        "Conservative management: activity modification, weight optimization.",
        "Low-impact exercise program (swimming, cycling, walking).",
        "PRN NSAIDs for symptomatic relief.",
        "Follow-up imaging in 12 months if symptoms persist or progress.",
    ],
    2: [
        "Structured physical therapy — quadriceps strengthening and ROM exercises.",
        "Intra-articular hyaluronate or PRP injection may be considered.",
        "Weight loss counseling if BMI ≥ 25.",
        "Reassess pain and function quarterly.",
        "Follow-up imaging in 6–12 months.",
    ],
    3: [
        "Multimodal pain management: scheduled NSAIDs, topical analgesics.",
        "Supervised physical therapy program.",
        "Consider intra-articular corticosteroid or genicular nerve block.",
        "Orthopaedic consultation recommended.",
        "Follow-up imaging in 3–6 months to monitor progression.",
    ],
    4: [
        "Urgent orthopaedic referral for total knee arthrothopy (TKA) evaluation.",
        "Pre-operative optimization: BMI, cardiac risk, dental clearance.",
        "Bridge therapy: intra-articular corticosteroid, opioids short-term if needed.",
        "Post-operative rehabilitation planning.",
    ],
}

MRI_SPECIFIC_RECOMMENDATIONS: dict[str, list[str]] = {
    "Meniscal Tear (Complex)": [
        "Orthopaedic referral — complex tears have higher surgical repair rates.",
        "Consider arthroscopic evaluation if mechanical symptoms present.",
    ],
    "Meniscal Tear (Flap)": [
        "Orthopaedic referral — displaced flap fragments often require arthroscopic intervention.",
    ],
    "Meniscal Tear (Extrusion)": [
        "Meniscal root repair may be indicated if concurrent ACL deficiency — orthopaedic review.",
    ],
    "Ligament Tear (Full Thickness/Complete Tear)": [
        "Urgent orthopaedic referral for ligament reconstruction assessment.",
        "Consider MRI follow-up after acute phase resolution.",
    ],
    "Cartilage Lesion (3)": [
        "Orthopaedic referral for cartilage restoration procedures (microfracture, OATS, ACI).",
    ],
}

# ── Limitations ────────────────────────────────────────────────────────────────

COMMON_LIMITATIONS: list[str] = [
    "AI-assisted analysis — findings should be confirmed by a qualified radiologist.",
    "Model confidence reflects statistical certainty, not clinical significance.",
]

XRAY_LIMITATIONS: dict[int, list[str]] = {
    0: ["Normal radiograph does not exclude early cartilage or soft-tissue pathology."],
    1: ["Early changes may be below radiographic detection threshold — MRI if symptoms persist."],
    2: ["Medial compartment narrowing may be positional — weight-bearing views recommended for confirmation."],
    3: ["Lateral and patellofemoral compartment assessment limited on AP views."],
    4: ["Severe OA with bone loss may underestimate cartilage status — MRI for surgical planning."],
}

MRI_LIMITATIONS: list[str] = [
    "MRI sensitivity varies with field strength, coil, and sequence protocol.",
    "Partial volume effects may overestimate lesion grade in thin cartilage regions.",
    "Artifact removal (MACS-Net) may introduce subtle texture changes — review raw and cleaned images.",
]

# ── Evidence descriptors ───────────────────────────────────────────────────────

XRAY_EVIDENCE: dict[int, list[str]] = {
    0: ["Preserved joint space bilaterally.", "No osteophyte or sclerosis visible."],
    1: ["Minimal marginal osteophyte at the tibial spines."],
    2: ["Definite osteophytes at femoral and tibial margins.", "Medial compartment joint space < 50% lateral."],
    3: ["Medial joint space ≤ 3 mm.", "Subchondral sclerosis at medial tibial plateau.", "Osteophytes at ≥ 3 margins."],
    4: ["Bone-on-bone contact medially.", "Subchondral cyst formation.", "Flattening of weight-bearing surface."],
}

MRI_EVIDENCE_MAP: dict[str, list[str]] = {
    "Meniscal Tear": ["Increased intrameniscal signal on proton-density or T2-weighted sequences."],
    "Ligament Tear": ["Altered ligament signal intensity and/or discontinuity on T2/PD sequences."],
    "Cartilage Lesion": ["Focal or diffuse cartilage signal change / thinning on T2/PD fat-sat sequences."],
    "Effusion": ["T2 hyperintensity within the suprapatellar recess and joint space."],
}

# ── Sources ────────────────────────────────────────────────────────────────────

SOURCES: list[str] = [
    "Kellgren JH, Lawrence JS. Radiological assessment of osteo-arthrosis. Ann Rheum Dis. 1957;16(4):494-502.",
    "Hunter DJ, et al. OARSI guidelines for the non-surgical management of knee osteoarthritis. Osteoarthritis Cartilage. 2019;27(3):349-362.",
    "Peterfy CG, et al. Whole-Organ Magnetic Resonance Imaging Score (WORMS) of the knee in osteoarthritis. Osteoarthritis Cartilage. 2004;12(3):177-190.",
    "International Cartilage Repair Society (ICRS) Cartilage Injury Evaluation Package. 2000.",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp_grade(grade: int) -> int:
    return max(0, min(4, int(grade)))


def _mri_findings_from_labels(positive_labels: list[str]) -> list[str]:
    """Generate MRI-specific findings from predicted SKM-TEA positive labels."""
    findings: list[str] = []
    seen_groups: set[str] = set()
    lookup: dict[str, list[str]] = {
        **MRI_MENISCAL_FINDINGS,
        **MRI_LIGAMENT_FINDINGS,
        **MRI_CARTILAGE_FINDINGS,
        **MRI_EFFUSION_FINDINGS,
    }
    for label in positive_labels:
        if label in lookup:
            findings.extend(lookup[label])
            if "Meniscal" in label:
                seen_groups.add("meniscal")
            elif "Ligament" in label:
                seen_groups.add("ligament")
            elif "Cartilage" in label:
                seen_groups.add("cartilage")
            elif "Effusion" in label:
                seen_groups.add("effusion")
    return findings


def _mri_evidence_from_labels(positive_labels: list[str]) -> list[str]:
    evidence: list[str] = []
    seen: set[str] = set()
    for label in positive_labels:
        for group, descs in MRI_EVIDENCE_MAP.items():
            if group in label and group not in seen:
                seen.add(group)
                evidence.extend(descs)
    return evidence


def _mri_recommendations_from_labels(positive_labels: list[str], grade: int) -> list[str]:
    recs: list[str] = []
    for label in positive_labels:
        if label in MRI_SPECIFIC_RECOMMENDATIONS:
            recs.extend(MRI_SPECIFIC_RECOMMENDATIONS[label])
    if not recs:
        recs = list(XRAY_RECOMMENDATIONS.get(grade, XRAY_RECOMMENDATIONS[0]))
    return recs


def _mri_limitations_from_labels(positive_labels: list[str], grade: int) -> list[str]:
    limits = list(COMMON_LIMITATIONS)
    limits.extend(MRI_LIMITATIONS)
    if grade >= 3:
        limits.append("Advanced degenerative changes may mask concurrent soft-tissue pathology.")
    return limits


def _xray_summary(grade: int, confidence: float) -> str:
    label = GRADE_LABELS[grade]
    return (
        f"Kellgren–Lawrence Grade {grade} ({label}) osteoarthritis "
        f"with {confidence:.1f}% ensemble confidence."
    )


def _mri_summary(grade: int, confidence: float, positive_labels: list[str]) -> str:
    label = GRADE_LABELS[grade]
    pathology_count = len(positive_labels)
    if pathology_count == 0:
        return (
            f"MRI analysis: KL Grade {grade} ({label}) — "
            f"no significant pathology above detection threshold ({confidence:.1f}% confidence)."
        )
    pathology_str = ", ".join(positive_labels[:3])
    suffix = f" (+{pathology_count - 3} more)" if pathology_count > 3 else ""
    return (
        f"MRI analysis: KL Grade {grade} ({label}) with {pathology_count} finding(s) — "
        f"{pathology_str}{suffix} ({confidence:.1f}% confidence)."
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_feedback(
    grade: int,
    modality: str,
    confidence: float,
    positive_labels: list[str] | None = None,
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate structured clinical feedback.

    Args:
        grade: KL grade (0–4).
        modality: "xray" or "mri".
        confidence: Model confidence (0–100).
        positive_labels: MRI positive pathology label names (ignored for X-ray).
        patient_context: Optional dict with age, gender, bmi, pain_level, etc.

    Returns:
        FeedbackBundle dict with summary, key_findings, recommendations,
        limitations, evidence, sources, grade_label, severity.
    """
    g = _clamp_grade(grade)
    positive_labels = positive_labels or []

    # ── Findings ────────────────────────────────────────────────────────────────
    if modality == "mri" and positive_labels:
        key_findings = _mri_findings_from_labels(positive_labels)
        # Fallback to grade-level findings if no specific pathology mapped
        if not key_findings:
            key_findings = XRAY_FINDINGS[g]  # KL findings as generic fallback
    else:
        key_findings = list(XRAY_FINDINGS[g])

    # ── Summary ─────────────────────────────────────────────────────────────────
    if modality == "mri":
        summary = _mri_summary(g, confidence, positive_labels)
    else:
        summary = _xray_summary(g, confidence)

    # Append patient context if available
    if patient_context:
        age = patient_context.get("age")
        gender = patient_context.get("gender", "")
        bmi = patient_context.get("bmi")
        if age and gender:
            ctx = f" Patient: {age}-year-old {str(gender).lower()}"
            if bmi:
                ctx += f", BMI {bmi}."
            else:
                ctx += "."
            summary += ctx

    # ── Recommendations ─────────────────────────────────────────────────────────
    if modality == "mri" and positive_labels:
        recommendations = _mri_recommendations_from_labels(positive_labels, g)
    else:
        recommendations = list(XRAY_RECOMMENDATIONS.get(g, XRAY_RECOMMENDATIONS[0]))

    # Patient-specific recommendation adjustments
    if patient_context and patient_context.get("bmi", 0) >= 25:
        bmi_rec = f"Weight management counseling recommended (current BMI: {patient_context['bmi']})."
        if bmi_rec not in recommendations:
            recommendations.append(bmi_rec)

    if patient_context and patient_context.get("pain_level", 0) >= 7:
        pain_rec = "Elevated pain level — consider multimodal analgesia and expedited specialist review."
        if pain_rec not in recommendations:
            recommendations.append(pain_rec)

    # ── Limitations ─────────────────────────────────────────────────────────────
    if modality == "mri":
        limitations = _mri_limitations_from_labels(positive_labels, g)
    else:
        limitations = list(COMMON_LIMITATIONS)
        limitations.extend(XRAY_LIMITATIONS.get(g, XRAY_LIMITATIONS[0]))

    # Confidence-based limitation
    if confidence < 70:
        limitations.insert(0, f"Low model confidence ({confidence:.1f}%) — manual review strongly recommended.")
    elif confidence < 85:
        limitations.insert(0, f"Moderate model confidence ({confidence:.1f}%) — clinical correlation advised.")

    # ── Evidence ────────────────────────────────────────────────────────────────
    if modality == "mri" and positive_labels:
        evidence = _mri_evidence_from_labels(positive_labels)
        if not evidence:
            evidence = list(XRAY_EVIDENCE.get(g, []))
    else:
        evidence = list(XRAY_EVIDENCE.get(g, []))

    return {
        "summary": summary,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "limitations": limitations,
        "evidence": evidence,
        "sources": list(SOURCES),
        "grade_label": GRADE_LABELS[g],
        "severity": SEVERITY_TAGS[g],
    }


def findings_for_grade(grade: int) -> list[str]:
    """Backward-compatible wrapper — returns plain findings list."""
    return list(XRAY_FINDINGS.get(_clamp_grade(grade), XRAY_FINDINGS[0]))


def mri_findings_for_labels(label_names: list[str]) -> list[str]:
    """Backward-compatible wrapper — returns MRI pathology findings."""
    if not label_names:
        return ["No significant pathology detected above model threshold."]
    return _mri_findings_from_labels(label_names) or [
        f"{name} identified on MRI." for name in label_names
    ]
