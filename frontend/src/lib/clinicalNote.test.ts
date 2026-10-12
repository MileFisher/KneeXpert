import { describe, expect, it } from "vitest";

import {
  applyNoteSignOff,
  approvedNoteText,
  approvedNotes,
  groundingBand,
  noteAttributionLine,
} from "./clinicalNote";
import type { PatientReport, SignedClinicalNote } from "@/data/patients";

function note(overrides: Partial<SignedClinicalNote> = {}): SignedClinicalNote {
  return {
    noteId: "n1",
    modality: "xray",
    generatedText: "Grade 3 change is present [1].",
    nextSteps: ["Correlate with symptoms [1]."],
    status: "pending",
    groundingScore: 0.7,
    gateAttempts: 1,
    fallbackReason: null,
    generatedBy: "meditron3-8b via Ollama",
    citations: [{ passageId: "kl-grading#04-00", source: "KL grading — Grade 3" }],
    generatedAt: "2026-10-12T10:00:00.000Z",
    ...overrides,
  };
}

function report(notes?: SignedClinicalNote[]): PatientReport {
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
    doctorConfirmed: false,
    doctorOverride: false,
    clinicalNotes: notes,
    updatedAt: "2026-10-12T10:00:00.000Z",
    version: 1,
  };
}

describe("applyNoteSignOff", () => {
  it("adds a note when the report has none", () => {
    const next = applyNoteSignOff(report(), note({ status: "accepted" }));
    expect(next.clinicalNotes).toHaveLength(1);
  });

  it("replaces the note for the same modality", () => {
    const first = note({ noteId: "n1", generatedText: "First." });
    const second = note({ noteId: "n2", generatedText: "Second." });
    const next = applyNoteSignOff(report([first]), second);
    expect(next.clinicalNotes).toHaveLength(1);
    expect(next.clinicalNotes?.[0].generatedText).toBe("Second.");
  });

  it("preserves a note for a different modality", () => {
    const xray = note({ noteId: "x", modality: "xray" });
    const mri = note({ noteId: "m", modality: "mri" });
    const next = applyNoteSignOff(report([xray]), mri);
    expect(next.clinicalNotes?.map((n) => n.modality).sort()).toEqual([
      "mri",
      "xray",
    ]);
  });

  it("bumps the version and timestamp so the change is traceable", () => {
    const next = applyNoteSignOff(report(), note({ status: "accepted" }));
    expect(next.version).toBe(2);
    expect(next.updatedAt).not.toBe(report().updatedAt);
  });

  it("does not mutate the report it was given", () => {
    const original = report();
    applyNoteSignOff(original, note());
    expect(original.clinicalNotes).toBeUndefined();
  });

  // The honesty invariant: a clinician's edit must never overwrite the record of
  // what the model actually wrote, or "reviewed by" becomes unfalsifiable.
  it("never mutates generatedText when a note is edited", () => {
    const original = note({ generatedText: "Model wrote this." });
    const edited = applyNoteSignOff(
      report([original]),
      note({
        generatedText: "Model wrote this.",
        reviewedText: "Clinician rewrote it.",
        status: "edited",
      }),
    );
    const stored = edited.clinicalNotes?.[0];
    expect(stored?.generatedText).toBe("Model wrote this.");
    expect(stored?.reviewedText).toBe("Clinician rewrote it.");
  });
});

describe("approvedNotes", () => {
  it("includes accepted and edited notes", () => {
    const notes = [
      note({ noteId: "a", modality: "xray", status: "accepted" }),
      note({ noteId: "b", modality: "mri", status: "edited" }),
    ];
    expect(approvedNotes(report(notes))).toHaveLength(2);
  });

  it("excludes pending and rejected notes", () => {
    const notes = [
      note({ noteId: "a", modality: "xray", status: "pending" }),
      note({ noteId: "b", modality: "mri", status: "rejected" }),
    ];
    expect(approvedNotes(report(notes))).toEqual([]);
  });

  it("tolerates a missing report or missing notes", () => {
    expect(approvedNotes(null)).toEqual([]);
    expect(approvedNotes(undefined)).toEqual([]);
    expect(approvedNotes(report())).toEqual([]);
  });
});

describe("approvedNoteText", () => {
  it("prefers the clinician's revision when there is one", () => {
    const text = approvedNoteText(
      note({ status: "edited", reviewedText: "Revised wording." }),
    );
    expect(text).toBe("Revised wording.");
  });

  it("falls back to the model text when accepted unchanged", () => {
    expect(approvedNoteText(note({ status: "accepted" }))).toBe(
      "Grade 3 change is present [1].",
    );
  });
});

describe("noteAttributionLine", () => {
  it("names the model, the reviewer and the date", () => {
    const line = noteAttributionLine(
      note({
        status: "accepted",
        reviewedBy: "Dr Tran",
        reviewedAt: "2026-10-12T11:30:00.000Z",
      }),
    );
    expect(line).toContain("meditron3-8b");
    expect(line).toContain("Dr Tran");
    expect(line).toContain("2026");
    expect(line.toLowerCase()).toContain("ai-assisted");
  });

  it("says unreviewed when there is no reviewer", () => {
    expect(noteAttributionLine(note()).toLowerCase()).toContain("not reviewed");
  });
});

describe("groundingBand", () => {
  // Shown instead of the raw figure: the threshold is provisional and fitted to
  // a handful of examples, so a bare "0.698" would read as calibrated when it is
  // not.
  it.each([
    [0.95, 0.6, "supported"],
    [0.6, 0.6, "supported"],
    [0.5, 0.6, "partial"],
    [0.3, 0.6, "weak"],
    [0, 0.6, "weak"],
  ])("maps %s at threshold %s to %s", (score, threshold, band) => {
    expect(groundingBand(score, threshold)).toBe(band);
  });

  // A note the gate accepted must never be labelled poorly supported, or the
  // UI contradicts the verdict that let it through.
  it("never labels a passing note as less than supported", () => {
    for (const threshold of [0.4, 0.6, 0.7, 0.85]) {
      expect(groundingBand(threshold, threshold)).toBe("supported");
      expect(groundingBand(threshold + 0.01, threshold)).toBe("supported");
    }
  });

  it("returns unknown when there is no score", () => {
    expect(groundingBand(null)).toBe("unknown");
  });
});
