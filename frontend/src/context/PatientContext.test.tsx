import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it } from "vitest";

import { PatientProvider, usePatients, type ConfirmDiagnosisPayload } from "./PatientContext";

function wrapper({ children }: { children: ReactNode }) {
  return <PatientProvider>{children}</PatientProvider>;
}

function payload(overrides: Partial<ConfirmDiagnosisPayload> = {}): ConfirmDiagnosisPayload {
  return {
    grade: 3,
    aiGrade: 3,
    aiConfidence: 88,
    findings: [],
    diagnosisSummary: "",
    modality: "xray",
    view: "AP",
    region: "Knee",
    inputFileName: "scan.png",
    modelUsed: "ensemble",
    ...overrides,
  };
}

beforeEach(() => {
  localStorage.clear();
});

describe("confirmDiagnosis", () => {
  // The bug: aiGrade was written from the final grade, so every override
  // erased what the AI had said and agreement could not be measured.
  it("keeps the AI's grade when the doctor overrides it", () => {
    const { result } = renderHook(() => usePatients(), { wrapper });
    const id = result.current.patients[0].id;

    act(() => {
      result.current.confirmDiagnosis(
        id,
        payload({ grade: 4, aiGrade: 2, doctorOverride: true }),
      );
    });

    const report = result.current.patients.find(p => p.id === id)?.report;
    expect(report?.aiGrade).toBe(2);
    expect(report?.finalGrade).toBe(4);
    expect(report?.doctorOverride).toBe(true);
  });

  it("does not drop a clinical note signed off before confirming", () => {
    const { result } = renderHook(() => usePatients(), { wrapper });
    const id = result.current.patients[0].id;

    act(() => {
      result.current.signOffClinicalNote(id, {
        noteId: "n1",
        modality: "combined",
        generatedText: "Combined note.",
        nextSteps: [],
        status: "accepted",
        groundingScore: 0.8,
        gateAttempts: 1,
        fallbackReason: null,
        generatedBy: "meditron3-8b via Ollama",
        citations: [],
        generatedAt: "2026-10-29T00:00:00.000Z",
      });
    });
    act(() => {
      result.current.confirmDiagnosis(id, payload());
    });

    const notes = result.current.patients.find(p => p.id === id)?.report?.clinicalNotes;
    expect(notes?.map(n => n.noteId)).toEqual(["n1"]);
  });
});
