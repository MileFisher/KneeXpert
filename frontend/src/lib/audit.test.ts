import { describe, expect, it } from "vitest";

import { buildAuditSummary, gradeSource, percent } from "./audit";
import type { Patient, PatientReport, SignedClinicalNote } from "@/data/patients";
import type { FusionResult } from "./diagnosticApi";

function report(overrides: Partial<PatientReport> = {}): PatientReport {
  return {
    aiGrade: 3,
    aiConfidence: 81,
    finalGrade: 3,
    modality: "xray",
    view: "AP",
    region: "Knee",
    inputFileName: "scan.png",
    findings: [],
    diagnosisSummary: "",
    modelUsed: "ensemble",
    doctorConfirmed: true,
    doctorOverride: false,
    updatedAt: "2026-10-29T00:00:00.000Z",
    version: 1,
    ...overrides,
  };
}

function patient(r: PatientReport | null): Patient {
  return { id: Math.random().toString(36), report: r } as unknown as Patient;
}

function note(overrides: Partial<SignedClinicalNote>): SignedClinicalNote {
  return {
    noteId: "n",
    modality: "xray",
    generatedText: "",
    nextSteps: [],
    status: "accepted",
    groundingScore: 0.8,
    gateAttempts: 1,
    fallbackReason: null,
    generatedBy: "meditron3-8b via Ollama",
    citations: [],
    generatedAt: "2026-10-29T00:00:00.000Z",
    ...overrides,
  };
}

const FUSION: FusionResult = {
  grade: 3,
  weighted_mean: 2.8,
  confidence: 75,
  agreement: "discordant",
  grade_gap: 2,
  weights: { xray: 0.6, mri: 0.4 },
  studies: [
    { modality: "xray", grade: 2, confidence: 90 },
    { modality: "mri", grade: 4, confidence: 60 },
  ],
  method: "confidence_weighted",
};

describe("buildAuditSummary", () => {
  it("measures one confirm and one override as 50% agreement", () => {
    const summary = buildAuditSummary([
      patient(report({ aiGrade: 3, finalGrade: 3 })),
      patient(report({ aiGrade: 2, finalGrade: 4, doctorOverride: true })),
    ]);
    expect(summary.agreement).toEqual({
      reviewed: 2,
      exact: 1,
      withinOne: 1,
      overridden: 1,
      meanAbsoluteDifference: 1,
    });
    expect(percent(summary.agreement.exact, summary.agreement.reviewed)).toBe(50);
  });

  it("ignores reports nobody has reviewed yet", () => {
    const summary = buildAuditSummary([
      patient(report({ doctorConfirmed: false, aiGrade: 0, finalGrade: 4 })),
    ]);
    expect(summary.agreement.reviewed).toBe(0);
    expect(summary.agreement.meanAbsoluteDifference).toBeNull();
  });

  it("files a fused grade under 'combined', with its discordance", () => {
    const fused = report({ modality: "xray", fusion: FUSION });
    expect(gradeSource(fused)).toBe("combined");
    const summary = buildAuditSummary([patient(fused), patient(report())]);
    expect(summary.agreementBySource.combined.reviewed).toBe(1);
    expect(summary.agreementBySource.xray.reviewed).toBe(1);
    expect(summary.discordance).toEqual({ fused: 1, discordant: 1, meanGap: 2 });
  });

  it("counts note outcomes per modality, combined included", () => {
    const summary = buildAuditSummary([
      patient(
        report({
          clinicalNotes: [
            note({ modality: "xray", status: "accepted", gateAttempts: 1 }),
            note({ modality: "mri", status: "rejected", gateAttempts: 3 }),
            note({ modality: "combined", status: "edited", gateAttempts: 2 }),
          ],
        }),
      ),
    ]);
    expect(summary.notes.total).toBe(3);
    expect(summary.notes.meanGateAttempts).toBe(2);
    expect(summary.notesByModality.combined.edited).toBe(1);
    expect(summary.notesByModality.mri.rejected).toBe(1);
  });

  it("is empty, not broken, with no patients", () => {
    const summary = buildAuditSummary([]);
    expect(summary.agreement.reviewed).toBe(0);
    expect(summary.notes.total).toBe(0);
    expect(summary.discordance.meanGap).toBeNull();
    expect(percent(0, 0)).toBeNull();
  });
});
