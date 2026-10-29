import { useQuery } from "@tanstack/react-query";

import { generateClinicalNote } from "@/lib/diagnosticApi";
import type {
  ClinicalNoteRequest,
  ClinicalNoteResponse,
} from "@/lib/diagnosticApi";

/**
 * Fetch the RAG-grounded clinical note for a diagnostic result.
 *
 * `useQuery` rather than `useMutation` despite the POST: the note is server
 * state derived from a stable key (this grade, this confidence, these labels),
 * not an action with side effects. That gets loading, error and refetch
 * handling for free and keeps the result cached when the user navigates away and
 * back.
 */

/** Stable, order-independent key fragment for the predicted labels. */
function labelsKey(req: ClinicalNoteRequest | null): string {
  if (!req?.positive_labels?.length) return "";
  return [...req.positive_labels]
    .map((label) => `${label.name}:${label.probability.toFixed(1)}`)
    .sort()
    .join("|");
}

/** Stable key fragment for a combined note's studies. */
function studiesKey(req: ClinicalNoteRequest | null): string {
  if (!req?.studies?.length) return "";
  return [...req.studies]
    .map(s => {
      const labels = (s.positive_labels ?? [])
        .map(label => `${label.name}@${label.probability.toFixed(1)}`)
        .sort()
        .join(",");
      return `${s.modality}:${s.grade}:${s.confidence.toFixed(1)}:${labels}`;
    })
    .sort()
    .join("|");
}

/**
 * Cache key for one patient's note.
 *
 * `patientId` is part of the key, not an afterthought: with `staleTime:
 * Infinity`, two patients who happen to share a modality, grade and confidence
 * would otherwise be served the same cached note — and the request carries
 * patient context, so the notes genuinely differ.
 */
export function clinicalNoteQueryKey(
  patientId: string | null,
  req: ClinicalNoteRequest | null,
) {
  return [
    "clinical-note",
    patientId,
    req?.modality ?? null,
    req?.grade ?? null,
    req?.confidence ?? null,
    req?.config ?? "rag_gate",
    labelsKey(req),
    studiesKey(req),
  ] as const;
}

export function useClinicalNote(
  patientId: string | null,
  req: ClinicalNoteRequest | null,
  enabled = true,
) {
  return useQuery<ClinicalNoteResponse>({
    queryKey: clinicalNoteQueryKey(patientId, req),
    queryFn: ({ signal }) => generateClinicalNote(req!, signal),
    enabled: enabled && req !== null && patientId !== null,
    // The backend already retries internally, escalating temperature and
    // widening retrieval. Retrying here would re-run that whole budget and
    // double the cost of metered inference for no new information.
    retry: false,
    // Generation is slow and deterministic for a given payload, so a refetch
    // would spend 10-60s to produce the same note.
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    gcTime: 30 * 60_000,
  });
}
