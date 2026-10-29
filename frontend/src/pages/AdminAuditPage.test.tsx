import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Patient, PatientReport } from "@/data/patients";

let patients: Patient[] = [];
vi.mock("@/context/PatientContext", () => ({
  usePatients: () => ({ patients }),
}));

// Recharts needs a measured container; jsdom has none.
vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 400, height: 200 }}>{children}</div>
    ),
  };
});

import AdminAuditPage from "./AdminAuditPage";

function withReport(r: Partial<PatientReport>): Patient {
  return {
    id: Math.random().toString(36),
    report: {
      aiGrade: 3,
      aiConfidence: 80,
      finalGrade: 3,
      modality: "xray",
      view: "AP",
      region: "Knee",
      inputFileName: "s.png",
      findings: [],
      diagnosisSummary: "",
      modelUsed: "ensemble",
      doctorConfirmed: true,
      doctorOverride: false,
      updatedAt: "2026-10-29",
      version: 1,
      ...r,
    },
  } as unknown as Patient;
}

describe("AdminAuditPage", () => {
  it("shows agreement from one confirm and one override", () => {
    patients = [
      withReport({ aiGrade: 3, finalGrade: 3 }),
      withReport({ aiGrade: 2, finalGrade: 4, doctorOverride: true }),
    ];
    render(<AdminAuditPage />);
    expect(screen.getByText("Performance Audit")).toBeInTheDocument();
    expect(screen.getByText("AI = final grade").previousSibling?.textContent).toBe("50%");
    expect(screen.getByText("Override rate").previousSibling?.textContent).toBe("50%");
  });

  it("says plainly that it changes nothing", () => {
    patients = [];
    render(<AdminAuditPage />);
    expect(screen.getByText(/Nothing here changes or retrains the models/)).toBeInTheDocument();
    expect(screen.getByText(/No clinical notes have been signed off/)).toBeInTheDocument();
  });
});
