import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Patient, SignedClinicalNote } from "@/data/patients";
import { mockPatients } from "@/data/patients";

/**
 * Covers decision 14 — a note reaches the exported record only after sign-off.
 *
 * This boundary was the one thing the planned Playwright spec would have caught
 * that component tests cannot: the pure helpers are tested in isolation, but
 * nothing otherwise verified that `generateReportPDF` actually consults them.
 * Rather than drive a browser, the jsPDF text calls are captured and inspected.
 */

const textCalls: string[] = [];

vi.mock("jspdf", () => {
  // A Proxy rather than a hand-listed stub: jsPDF's surface is large, and
  // enumerating it would make this test fail whenever the report starts using
  // another drawing call. Unknown members return a chainable no-op; only the
  // text calls are captured.
  const makeDoc = () => {
    const captured = {
      text(content: string | string[]) {
        textCalls.push(...(Array.isArray(content) ? content : [content]));
        return proxy;
      },
      splitTextToSize(content: string) {
        return [content];
      },
      getNumberOfPages: () => 1,
      internal: { pageSize: { getWidth: () => 210, getHeight: () => 297 } },
      lastAutoTable: { finalY: 100 },
    } as Record<string, unknown>;

    const proxy: unknown = new Proxy(captured, {
      get(target, prop) {
        if (prop in target) return target[prop as string];
        return () => proxy;
      },
    });
    return proxy;
  };

  return { default: function FakeDoc() { return makeDoc(); } };
});

vi.mock("jspdf-autotable", () => ({ default: vi.fn() }));

function note(overrides: Partial<SignedClinicalNote> = {}): SignedClinicalNote {
  return {
    noteId: "n1",
    modality: "xray",
    generatedText: "MODEL_SYNTHESIS_TEXT",
    nextSteps: ["NEXT_STEP_TEXT"],
    status: "accepted",
    groundingScore: 0.7,
    gateAttempts: 1,
    fallbackReason: null,
    generatedBy: "meditron3-8b via Ollama",
    citations: [],
    generatedAt: "2026-10-12T10:00:00.000Z",
    reviewedBy: "Dr Tran",
    reviewedAt: "2026-10-12T11:00:00.000Z",
    ...overrides,
  };
}

function patientWith(notes: SignedClinicalNote[]): Patient {
  const base = mockPatients[0];
  return {
    ...base,
    report: {
      aiGrade: 3,
      aiConfidence: 81,
      finalGrade: 3,
      modality: "xray",
      view: "AP",
      region: "Knee",
      inputFileName: "scan.png",
      findings: ["Existing deterministic finding."],
      diagnosisSummary: "Summary.",
      modelUsed: "ensemble",
      doctorConfirmed: true,
      doctorOverride: false,
      clinicalNotes: notes,
      updatedAt: "2026-10-12T11:00:00.000Z",
      version: 2,
    },
  };
}

async function renderPdf(patient: Patient): Promise<string> {
  textCalls.length = 0;
  const { generateReportPDF } = await import("./generateReportPDF");
  generateReportPDF(patient);
  return textCalls.join("\n");
}

beforeEach(() => {
  vi.resetModules();
});

describe("generateReportPDF — clinical note inclusion", () => {
  it("includes an accepted note with its attribution line", async () => {
    const output = await renderPdf(patientWith([note({ status: "accepted" })]));
    expect(output).toContain("MODEL_SYNTHESIS_TEXT");
    expect(output).toContain("NEXT_STEP_TEXT");
    expect(output).toContain("Dr Tran");
    expect(output).toContain("meditron3-8b");
    expect(output.toLowerCase()).toContain("ai-assisted");
  });

  it("prints the clinician's revision, not the model's original, when edited", async () => {
    const output = await renderPdf(
      patientWith([
        note({ status: "edited", reviewedText: "CLINICIAN_REVISION_TEXT" }),
      ]),
    );
    expect(output).toContain("CLINICIAN_REVISION_TEXT");
    expect(output).not.toContain("MODEL_SYNTHESIS_TEXT");
  });

  // Decision 14: the PDF is the record artifact, so this is where "reviewed
  // before it enters the record" is actually enforced.
  it("omits a rejected note entirely", async () => {
    const output = await renderPdf(patientWith([note({ status: "rejected" })]));
    expect(output).not.toContain("MODEL_SYNTHESIS_TEXT");
    expect(output).not.toContain("NEXT_STEP_TEXT");
  });

  it("omits a note that has not been reviewed", async () => {
    const output = await renderPdf(patientWith([note({ status: "pending" })]));
    expect(output).not.toContain("MODEL_SYNTHESIS_TEXT");
  });

  it("still produces a report when there are no notes at all", async () => {
    const output = await renderPdf(patientWith([]));
    expect(output).toContain("Existing deterministic finding.");
  });

  it("includes one note per modality", async () => {
    const output = await renderPdf(
      patientWith([
        note({ noteId: "a", modality: "xray", generatedText: "XRAY_NOTE" }),
        note({ noteId: "b", modality: "mri", generatedText: "MRI_NOTE" }),
      ]),
    );
    expect(output).toContain("XRAY_NOTE");
    expect(output).toContain("MRI_NOTE");
    expect(output).toContain("Clinical Note (XRAY)");
    expect(output).toContain("Clinical Note (MRI)");
  });
});
