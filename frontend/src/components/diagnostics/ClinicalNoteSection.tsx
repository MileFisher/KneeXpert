import { useCallback, useMemo } from "react";
import { toast } from "sonner";

import { ClinicalNotePanel } from "./ClinicalNotePanel";
import { useClinicalNote } from "@/hooks/useClinicalNote";
import { usePatients } from "@/context/PatientContext";
import type { Modality, Patient, SignedClinicalNote } from "@/data/patients";
import type { ClinicalNoteRequest, MriPredictResponse } from "@/lib/diagnosticApi";

/**
 * Wiring for the clinical-note panel: builds the request, runs the query, and
 * records a clinician's verdict.
 *
 * Kept out of `DiagnosticsPage` so that page gains only a mount point. The
 * panel itself stays presentational and therefore testable without a provider.
 */

type Props = {
  patient: Patient;
  modality: Modality;
  grade: number;
  confidence: number;
  /** Raw MRI response, when the active modality is MRI, for its label set. */
  mriData?: MriPredictResponse | null;
  /** Name recorded against the sign-off. */
  clinicianName?: string;
};

export function ClinicalNoteSection({
  patient,
  modality,
  grade,
  confidence,
  mriData,
  clinicianName = "Reviewing clinician",
}: Props) {
  const { signOffClinicalNote } = usePatients();

  const request = useMemo<ClinicalNoteRequest>(() => {
    const positive =
      modality === "mri"
        ? (mriData?.multilabel_predictions ?? [])
            .filter(label => label.predicted)
            .map(label => ({ name: label.name, probability: label.probability }))
        : [];
    return {
      modality,
      grade,
      confidence,
      positive_labels: positive,
      // Patient context lives in browser storage, so the backend has no access
      // to it and it must travel in the request body.
      patient_context: {
        age: patient.age,
        gender: patient.gender,
        bmi: patient.bmi,
      },
    };
  }, [modality, grade, confidence, mriData, patient.age, patient.gender, patient.bmi]);

  const query = useClinicalNote(request);
  const response = query.data ?? null;

  const existing = useMemo(
    () => patient.report?.clinicalNotes?.find(note => note.modality === modality),
    [patient.report?.clinicalNotes, modality],
  );

  const record = useCallback(
    (status: SignedClinicalNote["status"], extra: Partial<SignedClinicalNote>) => {
      const note = response?.clinical_note;
      if (!note) return;
      signOffClinicalNote(patient.id, {
        noteId: `${patient.id}-${modality}-${Date.now()}`,
        modality,
        generatedText: note.synthesis,
        nextSteps: note.next_steps,
        status,
        groundingScore: response?.grounding_score ?? null,
        gateAttempts: response?.gate_attempts ?? 0,
        fallbackReason: response?.fallback_reason ?? null,
        generatedBy: response?.generated_by ?? null,
        citations: (response?.retrieved_passages ?? []).map(passage => ({
          passageId: passage.passage_id,
          source: passage.source,
          url: passage.url ?? null,
        })),
        generatedAt: new Date().toISOString(),
        reviewedBy: clinicianName,
        reviewedAt: new Date().toISOString(),
        ...extra,
      });
    },
    [response, patient.id, modality, signOffClinicalNote, clinicianName],
  );

  return (
    <ClinicalNotePanel
      state={query.isPending ? "loading" : query.isError ? "error" : "ready"}
      response={response}
      error={query.error as Error | null}
      status={existing?.status ?? null}
      reviewedText={existing?.reviewedText}
      onAccept={() => {
        record("accepted", {});
        toast.success("Clinical note signed off", {
          description: "It will appear in the exported report.",
        });
      }}
      onEdit={text => {
        record("edited", { reviewedText: text });
        toast.success("Edited note signed off", {
          description: "The model's original wording is retained for audit.",
        });
      }}
      onReject={reason => {
        record("rejected", { rejectionReason: reason });
        toast.message("Clinical note rejected", {
          description: "It will not appear in the exported report.",
        });
      }}
      onRegenerate={() => {
        void query.refetch();
        toast.message("Regenerating clinical note…");
      }}
    />
  );
}
