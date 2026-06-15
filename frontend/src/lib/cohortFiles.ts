import type { Modality } from "@/data/patients";

export type CohortFilesMap = Map<string, Partial<Record<Modality, File>>>;

export function cohortFilesKey(patientId: string, modality: Modality): string {
  return `${patientId}:${modality}`;
}
