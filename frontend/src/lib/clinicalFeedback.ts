import type { Modality } from "@/data/patients";

/** Kellgren–Lawrence grade labels (KL 0–4). */
export const GRADE_LABELS: Record<number, string> = {
  0: "Normal",
  1: "Doubtful",
  2: "Minimal",
  3: "Moderate",
  4: "Severe",
};

/** Short narrative per KL grade — used in clinical interpretation summary. */
export const GRADE_NARRATIVE: Record<number, string> = {
  0: "No radiographic features of osteoarthritis. Joint space is preserved with no osteophyte formation.",
  1: "Doubtful narrowing of joint space and possible osteophytic lipping. Findings are minimal and may represent early degenerative change.",
  2: "Definite osteophytes and possible joint space narrowing. Mild osteoarthritis, with subchondral bone preserved.",
  3: "Multiple osteophytes, definite joint space narrowing, some sclerosis and possible deformity of bone contour. Moderate osteoarthritis.",
  4: "Large osteophytes, marked joint space narrowing, severe sclerosis and definite deformity of bone contour. Severe osteoarthritis.",
};

/**
 * X-ray findings by KL grade — aligned with backbone `ensemble.FINDINGS_BY_GRADE`.
 */
export const XRAY_FINDINGS_BY_GRADE: Record<number, readonly string[]> = {
  0: ["No radiographic features of osteoarthritis.", "Joint space preserved."],
  1: ["Doubtful joint space narrowing.", "Possible early osteophytic change."],
  2: [
    "Definite osteophytes identified.",
    "Possible joint space narrowing in medial compartment.",
    "Mild subchondral changes.",
  ],
  3: [
    "Joint space narrowing (medial compartment).",
    "Osteophyte formation (tibial plateau).",
    "Subchondral sclerosis detected.",
  ],
  4: [
    "Marked joint space narrowing.",
    "Large osteophytes and subchondral sclerosis.",
    "Definite bone contour deformity.",
  ],
};

/** MRI findings by KL-aligned grade (clinical feedback placeholder until MRI model is live). */
export const MRI_FINDINGS_BY_GRADE: Record<number, readonly string[]> = {
  0: [
    "Cartilage thickness and signal preserved throughout weight-bearing surfaces.",
    "Menisci intact without degenerative tear or extrusion.",
    "No significant joint effusion.",
  ],
  1: [
    "Mild cartilage signal heterogeneity without focal full-thickness defect.",
    "Possible early meniscal signal change; surfaces grossly intact.",
    "Trace effusion within physiological range.",
  ],
  2: [
    "Focal cartilage thinning over the medial femoral condyle.",
    "Mild meniscal degeneration without maceration.",
    "Minimal joint effusion.",
  ],
  3: [
    "Partial- to full-thickness cartilage loss with subchondral marrow signal change.",
    "Meniscal degeneration with surface fraying.",
    "Moderate joint effusion.",
  ],
  4: [
    "Extensive cartilage loss and subchondral bone exposure.",
    "Advanced meniscal maceration or extrusion.",
    "Marked effusion and synovial thickening.",
  ],
};

export function clampGrade(grade: number): number {
  return Math.min(4, Math.max(0, Math.round(grade)));
}

export function getFindingsForGrade(modality: Modality, grade: number): string[] {
  const g = clampGrade(grade);
  const table = modality === "xray" ? XRAY_FINDINGS_BY_GRADE : MRI_FINDINGS_BY_GRADE;
  return [...(table[g] ?? table[0])];
}

export function getGradeNarrative(grade: number): string {
  const g = clampGrade(grade);
  return GRADE_NARRATIVE[g] ?? GRADE_NARRATIVE[0];
}

export function getRecommendationForGrade(grade: number): string {
  const g = clampGrade(grade);
  if (g <= 1) {
    return "Conservative management: weight optimization, low-impact exercise, NSAIDs as needed. Re-image in 12 months if symptoms persist.";
  }
  if (g === 2) {
    return "Structured physical therapy, intra-articular hyaluronate may be considered. Reassess pain/function quarterly.";
  }
  if (g === 3) {
    return "Multimodal pain management, supervised PT, consider intra-articular corticosteroid or genicular nerve block. Orthopaedic consult recommended.";
  }
  return "Refer to orthopaedic surgery for evaluation of total knee arthroplasty. Pre-operative optimization (BMI, cardiac, dental clearance) advised.";
}

export function getGradeLabel(grade: number): string {
  return GRADE_LABELS[clampGrade(grade)] ?? "Unknown";
}
