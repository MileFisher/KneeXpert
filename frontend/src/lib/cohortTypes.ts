import type { Modality } from "@/data/patients";

export type ModalityUpload = { fileName: string; previewUrl: string | null };
export type CohortInputEntry = Partial<Record<Modality, ModalityUpload>>;
