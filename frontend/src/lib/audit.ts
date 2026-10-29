import type {
  NoteReviewStatus,
  Patient,
  PatientReport,
  SignedClinicalNote,
} from "@/data/patients";
import type { NoteModality } from "@/lib/diagnosticApi";

/**
 * Performance audit, computed from what clinicians have already recorded.
 *
 * Read-only by design: it reports how the AI's output fared under review and
 * feeds nothing back into the models. Everything comes from the patient records
 * in browser storage, so the figures cover this browser's patients only, and
 * the seeded demo patients count as reviewed.
 */

/** How a report's grade was produced: one study, or the orchestrator's fusion. */
export type GradeSource = "xray" | "mri" | "combined";

export type GradeAgreement = {
  /** Reports a clinician confirmed or overrode. */
  reviewed: number;
  /** AI grade equal to the final grade. */
  exact: number;
  /** AI grade within one KL grade of the final grade. */
  withinOne: number;
  overridden: number;
  /** Mean |AI grade − final grade|; null when nothing was reviewed. */
  meanAbsoluteDifference: number | null;
};

export type NoteOutcomes = Record<NoteReviewStatus, number> & {
  total: number;
  /** Mean generation attempts per signed note; null when there are none. */
  meanGateAttempts: number | null;
};

export type Discordance = {
  /** Reports whose grade the orchestrator fused from an X-ray and an MRI. */
  fused: number;
  discordant: number;
  /** Mean grade gap across fused reports; null when there are none. */
  meanGap: number | null;
};

export type AuditSummary = {
  agreement: GradeAgreement;
  agreementBySource: Record<GradeSource, GradeAgreement>;
  notes: NoteOutcomes;
  notesByModality: Record<NoteModality, NoteOutcomes>;
  discordance: Discordance;
};

export function gradeSource(report: PatientReport): GradeSource {
  return (report.fusion?.studies.length ?? 0) > 1 ? "combined" : report.modality;
}

function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function agreementOf(reports: PatientReport[]): GradeAgreement {
  const reviewed = reports.filter(r => r.doctorConfirmed);
  const gaps = reviewed.map(r => Math.abs(r.aiGrade - r.finalGrade));
  return {
    reviewed: reviewed.length,
    exact: gaps.filter(g => g === 0).length,
    withinOne: gaps.filter(g => g <= 1).length,
    overridden: reviewed.filter(r => r.doctorOverride).length,
    meanAbsoluteDifference: mean(gaps),
  };
}

function outcomesOf(notes: SignedClinicalNote[]): NoteOutcomes {
  const count = (status: NoteReviewStatus) => notes.filter(n => n.status === status).length;
  return {
    total: notes.length,
    pending: count("pending"),
    accepted: count("accepted"),
    edited: count("edited"),
    rejected: count("rejected"),
    meanGateAttempts: mean(notes.map(n => n.gateAttempts)),
  };
}

export function buildAuditSummary(patients: Patient[]): AuditSummary {
  const reports = patients.flatMap(p => (p.report ? [p.report] : []));
  const notes = reports.flatMap(r => r.clinicalNotes ?? []);
  const fused = reports.filter(r => gradeSource(r) === "combined");
  const gaps = fused.map(r => r.fusion?.grade_gap ?? 0);

  const sources: GradeSource[] = ["xray", "mri", "combined"];
  const noteModalities: NoteModality[] = ["xray", "mri", "combined"];

  return {
    agreement: agreementOf(reports),
    agreementBySource: Object.fromEntries(
      sources.map(source => [source, agreementOf(reports.filter(r => gradeSource(r) === source))]),
    ) as Record<GradeSource, GradeAgreement>,
    notes: outcomesOf(notes),
    notesByModality: Object.fromEntries(
      noteModalities.map(m => [m, outcomesOf(notes.filter(n => n.modality === m))]),
    ) as Record<NoteModality, NoteOutcomes>,
    discordance: {
      fused: fused.length,
      discordant: fused.filter(r => r.fusion?.agreement === "discordant").length,
      meanGap: mean(gaps),
    },
  };
}

/** A share as a whole-number percentage, or null when there is no denominator. */
export function percent(part: number, whole: number): number | null {
  return whole > 0 ? Math.round((part / whole) * 100) : null;
}
