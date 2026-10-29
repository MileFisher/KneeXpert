import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Patient } from "@/data/patients";
import type { ClinicalNoteRequest, StudyResult } from "@/lib/diagnosticApi";

const requests: (ClinicalNoteRequest | null)[] = [];

vi.mock("@/hooks/useClinicalNote", () => ({
  useClinicalNote: (_patientId: string, req: ClinicalNoteRequest | null) => {
    requests.push(req);
    return { isPending: true, isFetching: false, isError: false, data: undefined };
  },
}));
vi.mock("@/context/PatientContext", () => ({
  usePatients: () => ({ signOffClinicalNote: vi.fn() }),
}));

import { ClinicalNoteSection } from "./ClinicalNoteSection";

const patient = {
  id: "PT-1",
  age: 65,
  gender: "Male",
  bmi: 27.3,
  painLevel: 6,
  history: "",
  symptoms: "",
  report: null,
} as unknown as Patient;

const studies: StudyResult[] = [
  { modality: "xray", grade: 2, confidence: 90, positive_labels: [] },
  {
    modality: "mri",
    grade: 4,
    confidence: 60,
    positive_labels: [{ name: "Effusion", probability: 86 }],
  },
];

describe("ClinicalNoteSection", () => {
  it("sends both studies for a combined note", () => {
    requests.length = 0;
    render(
      <ClinicalNoteSection
        patient={patient}
        modality="combined"
        grade={3}
        confidence={75}
        studies={studies}
      />,
    );
    const req = requests[requests.length - 1];
    expect(req?.modality).toBe("combined");
    expect(req?.studies).toEqual(studies);
  });

  it("sends no studies for a single-scan note", () => {
    requests.length = 0;
    render(
      <ClinicalNoteSection
        patient={patient}
        modality="xray"
        grade={2}
        confidence={90}
        studies={studies}
      />,
    );
    expect(requests[requests.length - 1]?.studies).toBeUndefined();
  });
});
