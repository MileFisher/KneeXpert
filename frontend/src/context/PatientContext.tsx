import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  mockPatients,
  type Modality,
  type ModalityReportSnapshot,
  type Patient,
  type PatientReport,
  type ReportModelSnapshot,
} from "@/data/patients";
import { loadPatients, savePatients } from "@/lib/patientStorage";

export type ConfirmDiagnosisPayload = {
  grade: number;
  aiConfidence: number;
  findings: string[];
  diagnosisSummary: string;
  modality: Modality;
  view: string;
  region: string;
  inputFileName: string;
  modelUsed: string;
  inputImageDataUrl?: string | null;
  ensembleGradcamDataUrl?: string | null;
  modelResults?: ReportModelSnapshot[];
  modalitySnapshots?: ModalityReportSnapshot[];
  doctorOverride?: boolean;
  overrideNotes?: string;
};

export type ApplyAnalysisPayload = Omit<
  ConfirmDiagnosisPayload,
  "doctorOverride" | "overrideNotes"
> & {
  modalitySnapshots?: ModalityReportSnapshot[];
};

function buildInitialReport(p: Omit<Patient, "report"> & { report?: PatientReport | null }): PatientReport | null {
  const scan = p.scans.find(s => s.grade != null) ?? p.scans[0];
  if (!scan || p.grade == null || p.aiConfidence == null) return null;
  return {
    aiGrade: p.grade,
    aiConfidence: p.aiConfidence,
    finalGrade: p.grade,
    modality: scan.modality,
    view: scan.view ?? "",
    region: scan.region,
    inputFileName: `${scan.region}${scan.view ? ` · ${scan.view}` : ""}`,
    findings: [],
    diagnosisSummary: "",
    modelUsed: scan.modelUsed,
    doctorConfirmed: p.status === "confirmed",
    doctorOverride: false,
    updatedAt: p.lastVisit,
    version: p.status === "confirmed" ? 1 : 0,
  };
}

function hydratePatient(raw: Omit<Patient, "report"> & { report?: PatientReport | null }): Patient {
  return {
    ...raw,
    report: raw.report ?? buildInitialReport(raw),
  };
}

interface PatientContextValue {
  patients: Patient[];
  getPatient: (id: string) => Patient | undefined;
  applyAnalysisResult: (patientId: string, payload: ApplyAnalysisPayload) => void;
  confirmDiagnosis: (patientId: string, payload: ConfirmDiagnosisPayload) => void;
}

const PatientContext = createContext<PatientContextValue | null>(null);

export function PatientProvider({ children }: { children: ReactNode }) {
  const [patients, setPatients] = useState<Patient[]>(() => {
    const stored = loadPatients();
    return (stored ?? mockPatients).map(hydratePatient);
  });

  useEffect(() => {
    savePatients(patients);
  }, [patients]);

  const getPatient = useCallback(
    (id: string) => patients.find(p => p.id === id),
    [patients],
  );

  const applyAnalysisResult = useCallback((patientId: string, payload: ApplyAnalysisPayload) => {
    const today = new Date().toISOString().slice(0, 10);
    setPatients(prev =>
      prev.map(p => {
        if (p.id !== patientId) return p;
        const report: PatientReport = {
          aiGrade: payload.grade,
          aiConfidence: payload.aiConfidence,
          finalGrade: payload.grade,
          modality: payload.modality,
          view: payload.view,
          region: payload.region,
          inputFileName: payload.inputFileName,
          findings: payload.findings,
          diagnosisSummary: payload.diagnosisSummary,
          modelUsed: payload.modelUsed,
          inputImageDataUrl: payload.inputImageDataUrl,
          ensembleGradcamDataUrl: payload.ensembleGradcamDataUrl,
          modelResults: payload.modelResults,
          modalitySnapshots: payload.modalitySnapshots,
          doctorConfirmed: false,
          doctorOverride: false,
          updatedAt: today,
          version: Math.max(1, (p.report?.version ?? 0)),
        };
        return {
          ...p,
          grade: payload.grade,
          aiConfidence: payload.aiConfidence,
          status: payload.aiConfidence < 80 ? "flagged" : "analyzed",
          lastVisit: today,
          report,
          timeline: [
            {
              date: today,
              type: "diagnosis",
              summary: `AI Classification: Grade ${payload.grade} OA (${payload.aiConfidence.toFixed(1)}%) — pending review`,
              grade: payload.grade,
              confidence: payload.aiConfidence,
            },
            ...p.timeline.filter(t => t.type !== "diagnosis" || !t.summary.startsWith("AI Classification")),
          ],
        };
      }),
    );
  }, []);

  const confirmDiagnosis = useCallback((patientId: string, payload: ConfirmDiagnosisPayload) => {
    const today = new Date().toISOString().slice(0, 10);
    setPatients(prev =>
      prev.map(p => {
        if (p.id !== patientId) return p;
        const finalGrade = payload.doctorOverride ? payload.grade : payload.grade;
        const nextVersion = (p.report?.version ?? 0) + 1;
        const report: PatientReport = {
          aiGrade: payload.grade,
          aiConfidence: payload.aiConfidence,
          finalGrade,
          modality: payload.modality,
          view: payload.view,
          region: payload.region,
          inputFileName: payload.inputFileName,
          findings: payload.findings,
          diagnosisSummary: payload.diagnosisSummary,
          modelUsed: payload.modelUsed,
          inputImageDataUrl: payload.inputImageDataUrl ?? p.report?.inputImageDataUrl,
          ensembleGradcamDataUrl: payload.ensembleGradcamDataUrl ?? p.report?.ensembleGradcamDataUrl,
          modelResults: payload.modelResults ?? p.report?.modelResults,
          modalitySnapshots: payload.modalitySnapshots ?? p.report?.modalitySnapshots,
          doctorConfirmed: true,
          doctorOverride: payload.doctorOverride ?? false,
          overrideNotes: payload.overrideNotes,
          updatedAt: today,
          version: nextVersion,
        };
        const summary = payload.doctorOverride
          ? `Report v${nextVersion}: Grade ${finalGrade} OA (${payload.aiConfidence.toFixed(1)}%) — doctor override`
          : `Report v${nextVersion}: Grade ${finalGrade} OA (${payload.aiConfidence.toFixed(1)}%) — confirmed`;
        return {
          ...p,
          grade: finalGrade,
          aiConfidence: payload.aiConfidence,
          status: payload.aiConfidence < 80 ? "flagged" : "confirmed",
          lastVisit: today,
          report,
          timeline: [
            { date: today, type: "report", summary, grade: finalGrade, confidence: payload.aiConfidence },
            ...p.timeline,
          ],
        };
      }),
    );
  }, []);

  const value = useMemo(
    () => ({ patients, getPatient, applyAnalysisResult, confirmDiagnosis }),
    [patients, getPatient, applyAnalysisResult, confirmDiagnosis],
  );

  return <PatientContext.Provider value={value}>{children}</PatientContext.Provider>;
}

export function usePatients() {
  const ctx = useContext(PatientContext);
  if (!ctx) throw new Error("usePatients must be used within PatientProvider");
  return ctx;
}
