import type {
  NoteReviewStatus,
  PatientReport,
  SignedClinicalNote,
} from "@/data/patients";
import type { ClinicalNoteResponse } from "@/lib/diagnosticApi";

/**
 * Pure helpers for the AI clinical note and its physician sign-off.
 *
 * Kept out of components so the rules that matter — what counts as approved,
 * what reaches the exported record, and the fact that a clinician's edit never
 * overwrites the model's own words — are testable without rendering anything.
 */

/** Statuses whose text may enter the patient record. */
const APPROVED: readonly NoteReviewStatus[] = ["accepted", "edited"];

export type GroundingBand = "supported" | "partial" | "weak" | "unknown";

/**
 * Qualitative grounding label, relative to the threshold the note was judged at.
 *
 * Deliberately not the raw score. The groundedness threshold is currently
 * provisional — fitted to a handful of examples pending a proper sweep — so
 * showing a clinician "0.698" would present a number as calibrated when it is
 * not. The figure stays available behind a disclosure for anyone who wants it.
 *
 * Banded against `threshold` rather than fixed cut-offs, because a note that the
 * gate accepted must not then be labelled as poorly supported: with a gate at
 * 0.60, hardcoded bands at 0.70 made a passing note read "partially supported",
 * so the UI contradicted the system's own verdict.
 */
export function groundingBand(
  score: number | null,
  threshold = 0.6,
): GroundingBand {
  if (score === null || Number.isNaN(score)) return "unknown";
  if (score >= threshold) return "supported";
  // Within three-quarters of the bar: short of it, but not unsupported.
  if (score >= threshold * 0.75) return "partial";
  return "weak";
}

const BAND_ORDER: readonly GroundingBand[] = ["weak", "partial", "supported"];

/**
 * The band for a whole note: the weaker of its two parts, each judged against
 * its own threshold.
 *
 * The gate passes a note only when both parts clear their bars, so a passing
 * note always bands as supported here. Banding the pooled score instead would
 * break that once the recommendation threshold sits below the synthesis one: a
 * passing note can pool below the synthesis bar. Falls back to the pooled score
 * when the backend predates per-part scoring.
 */
export function noteGroundingBand(
  response: Pick<
    ClinicalNoteResponse,
    | "grounding_score"
    | "gate_threshold"
    | "synthesis_groundedness"
    | "next_steps_groundedness"
    | "next_steps_threshold"
  >,
): GroundingBand {
  const synthesis = response.synthesis_groundedness;
  if (synthesis == null) {
    return groundingBand(response.grounding_score ?? null, response.gate_threshold);
  }
  const bands = [groundingBand(synthesis, response.gate_threshold)];
  if (response.next_steps_groundedness != null) {
    bands.push(
      groundingBand(
        response.next_steps_groundedness,
        response.next_steps_threshold ?? response.gate_threshold,
      ),
    );
  }
  return bands.reduce((worst, band) =>
    BAND_ORDER.indexOf(band) < BAND_ORDER.indexOf(worst) ? band : worst,
  );
}

/**
 * Record a clinician's verdict on a note, returning a new report.
 *
 * Replaces any existing note for the same modality and leaves other modalities
 * untouched: a note belongs to a study, not to the patient as a whole.
 */
export function applyNoteSignOff(
  report: PatientReport,
  note: SignedClinicalNote,
): PatientReport {
  const others = (report.clinicalNotes ?? []).filter(
    (existing) => existing.modality !== note.modality,
  );
  return {
    ...report,
    clinicalNotes: [...others, note],
    updatedAt: new Date().toISOString(),
    version: report.version + 1,
  };
}

/** Notes a clinician has approved, and which may therefore be exported. */
export function approvedNotes(
  report: PatientReport | null | undefined,
): SignedClinicalNote[] {
  return (report?.clinicalNotes ?? []).filter((note) =>
    APPROVED.includes(note.status),
  );
}

/** The text a clinician actually signed off: their revision, else the model's. */
export function approvedNoteText(note: SignedClinicalNote): string {
  return note.reviewedText?.trim() ? note.reviewedText : note.generatedText;
}

/**
 * Provenance line printed beneath an exported note.
 *
 * Names the model and the reviewer so responsibility for the text is
 * unambiguous in the record, and says so plainly when nobody has reviewed it.
 */
/** Heading label for a note's modality in the exported record. */
export function noteModalityLabel(modality: SignedClinicalNote["modality"]): string {
  if (modality === "combined") return "Combined X-ray + MRI";
  return modality === "xray" ? "X-ray" : "MRI";
}

export function noteAttributionLine(note: SignedClinicalNote): string {
  const model = note.generatedBy ?? "an AI model";
  if (!note.reviewedBy) {
    return `AI-assisted (${model}) — not reviewed by a clinician.`;
  }
  const when = note.reviewedAt
    ? new Date(note.reviewedAt).toLocaleDateString("en-GB", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : "an unrecorded date";
  return `AI-assisted (${model}), reviewed by ${note.reviewedBy} on ${when}.`;
}
