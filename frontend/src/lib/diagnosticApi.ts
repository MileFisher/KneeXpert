const API_BASE = import.meta.env.VITE_BACKBONE_URL ?? "http://localhost:9000";

export type UploadProgressHandler = (percent: number) => void;

export type XrayModelResult = {
  model_id: string;
  display_name: string;
  family?: string;
  variant?: string;
  predicted_class: string;
  grade: number;
  confidence: number;
  gradcam_base64: string | null;
  class_probabilities: Record<string, number>;
};

export type XrayModelCatalogEntry = {
  id: string;
  display_name: string;
  family: string;
  variant: string;
  weights_file: string;
  /** Multiply-accumulate ops — the figure torchvision/timm papers quote. */
  gmacs?: number;
  /** True floating-point ops = 2 x gmacs. */
  gflops?: number;
  params_m?: number;
};

/** Structured clinical feedback from the backend feedback engine. */
export type FeedbackBundle = {
  summary: string;
  key_findings: string[];
  recommendations: string[];
  limitations: string[];
  evidence: string[];
  sources: string[];
  grade_label: string;
  severity: string;
};

/**
 * Compute cost for a single model.
 *
 * `gmacs` is the source-of-truth count (multiply-accumulates, the fvcore /
 * torchvision convention that papers usually mislabel as "FLOPs"); `gflops`
 * is the true floating-point count, `2 x gmacs`.
 */
export type ModelFlopsEntry = {
  gmacs: number;
  gflops: number;
  params_m: number;
  /** "image" for X-ray models, "forward_pass" for MRI pipeline stages. */
  per: "image" | "forward_pass";
  /** Where the figure came from: a live fvcore count or a published estimate. */
  method: "fvcore" | "architecture_estimate";
  /** MRI only — MACS-Net runs 3x per sampled slice (2.5D stacking). */
  calls_per_sampled_slice?: number;
  /**
   * Set when fvcore's count was incomplete: either it was discarded as an
   * undercount (`method` stays "architecture_estimate") or it was adopted but
   * some ops had no handler, making it a lower bound.
   */
  measurement_warning?: string;
  /** The discarded fvcore figure, when one failed the plausibility check. */
  rejected_fvcore_gmacs?: number;
  /**
   * Why no measurement happened at all — missing checkpoint, missing fvcore,
   * or a tracing failure. Distinct from `measurement_warning`, which reports a
   * measurement that ran and was then rejected.
   */
  profiling_error?: string;
};

/** Combined cost of running a set of X-ray models over one image. */
export type XrayEnsembleCost = {
  gmacs: number;
  gflops: number;
  params_m: number;
  model_count: number;
  model_ids: string[];
};

/** Cost of one sampled MRI slice: 3 x MACS-Net + 1 x DeiT. */
export type MriPerSliceCost = {
  gmacs: number;
  gflops: number;
  macs_net_calls: number;
  deit_calls: number;
};

/** Study-level MRI cost = per-sampled-slice cost x slices processed. */
export type MriStudyCost = {
  gmacs: number;
  gflops: number;
  sampled_slices: number;
  /** True when the backend fell back to MAX_SAMPLES_PER_STUDY. */
  is_upper_bound: boolean;
  params_m?: number;
};

/**
 * `model_flops` payload: per-model entries keyed by model id, plus the two
 * MRI aggregate keys and a units note. Read per-model entries through
 * {@link getModelFlops} so the aggregate keys can't be mistaken for a model.
 */
export type ModelFlopsMap = {
  unit_note?: string;
  /** X-ray: combined cost of the models that actually ran (2+ models only). */
  ensemble_total?: XrayEnsembleCost;
  pipeline_per_sampled_slice?: MriPerSliceCost;
  pipeline_total?: MriStudyCost;
  [modelId: string]:
    | ModelFlopsEntry
    | XrayEnsembleCost
    | MriPerSliceCost
    | MriStudyCost
    | string
    | undefined;
};

/** Look up one model's compute cost, narrowing past the aggregate keys. */
export function getModelFlops(
  map: ModelFlopsMap | undefined,
  modelId: string,
): ModelFlopsEntry | undefined {
  const entry = map?.[modelId];
  if (entry && typeof entry === "object" && "per" in entry) {
    return entry as ModelFlopsEntry;
  }
  return undefined;
}

export type XrayPredictResponse = {
  filename: string;
  grade: number;
  confidence: number;
  is_reliable: boolean;
  findings: string[];
  class_probabilities: Record<string, number>;
  individual_results: Record<string, XrayModelResult>;
  gradcam_base64: string | null;
  models_used: string[];
  model_count?: number;
  ensemble_display_name?: string;
  feedback?: FeedbackBundle;
  model_flops?: ModelFlopsMap;
};

export type MriLabelPrediction = {
  name: string;
  probability: number;
  predicted: boolean;
};

export type MriVolumeMeta = {
  shape: number[];
  slice_axis: number;
  slice_axis_label: string;
  num_slices: number;
  in_plane_shape: number[];
  orientation_axcodes: string[];
  plane_description: string;
};

export type MriSliceGalleryEntry = {
  slice_idx: number;
  slice_axis: number;
  raw_base64: string | null;
  cleaned_base64: string | null;
  artifact_map_base64: string | null;
  gradcam_base64: string | null;
  predicted_labels: string[];
  probabilities: number[];
};

export type MriPreview = {
  center_slice_idx: number;
  slice_axis?: number;
  raw_base64: string | null;
  cleaned_base64: string | null;
  artifact_map_base64?: string | null;
  gradcam_base64?: string | null;
};

export type MriPredictResponse = {
  filename: string;
  pipeline_mode: string;
  artifact_removal: string;
  classifier: string;
  volume_shape: number[];
  volume_meta?: MriVolumeMeta;
  slices_processed: number;
  slice_indices: number[];
  gallery_slice_indices?: number[];
  primary_slice_idx?: number;
  grade: number;
  confidence: number;
  is_reliable: boolean;
  threshold: number;
  findings: string[];
  pathology_findings: string[];
  kl_findings: string[];
  multilabel_predictions: MriLabelPrediction[];
  models_used: string[];
  preview: MriPreview;
  slice_gallery?: MriSliceGalleryEntry[];
  gradcam_base64?: string | null;
  artifact_map_base64?: string | null;
  sample_mode?: boolean;
  category_feedback?: string[];
  ground_truth_labels?: string[];
  ground_truth_feedback?: string[];
  skm_tea_categories?: { id: number; name: string }[];
  feedback?: FeedbackBundle;
  model_flops?: ModelFlopsMap;
};

export type BackboneHealth = {
  status: string;
  xray_models_available: string[];
  model_catalog: XrayModelCatalogEntry[];
  default_ensemble: string[];
  mri_models_available?: Record<string, boolean>;
  mri_pipeline_ready?: boolean;
  mri_sample_available?: boolean;
  mri_sample_filename?: string | null;
  model_flops?: {
    unit_note: string;
    xray_models: Record<string, ModelFlopsEntry & { family: string }>;
    mri_models: Record<string, ModelFlopsEntry & { display_name: string }>;
    totals: {
      xray_mean_per_model: { gmacs: number; gflops: number };
      /** The recommended 3-model ensemble (matches `default_ensemble`). */
      default_xray_ensemble: XrayEnsembleCost;
      /** Every configured model — what `model_names="all"` runs. */
      xray_ensemble_all: { gmacs: number; gflops: number; model_count: number };
      mri_per_sampled_slice: MriPerSliceCost;
      mri_per_study_max: MriStudyCost;
    };
  };

  /**
   * Clinical-note layer readiness. All optional: a backend without the layer
   * deployed simply omits them.
   */
  /** An on-disk vector index exists. False means `python -m rag.ingest` has not run. */
  rag_index_ready?: boolean;
  rag_index_path?: string;
  /** Passages in the committed corpus. */
  rag_chunk_count?: number;
  /**
   * Whether each local model is already in the HuggingFace cache. False means the
   * first note request will stall on a download — worth checking before a demo.
   */
  rag_models_cached?: { embedder: boolean; nli: boolean };
  rag_models_ready?: boolean;
  /** The generation host answered. Probed with a short timeout and cached server-side. */
  llm_available?: boolean;
  /** Configured model tag, present whether or not the host is reachable. */
  llm_model?: string;
  /** The host is reachable AND has `llm_model` pulled. */
  llm_model_present?: boolean;
  llm_base_url?: string;
  /** Why the probe failed, when it did. Null when the host answered. */
  llm_error?: string | null;
};

// ── Clinical note (RAG-grounded) ─────────────────────────────────────────────
// Mirrors backend/rag/schemas.py. Additive: FeedbackBundle above is unchanged.

/** Imaging modality, as the backend names it. */
/** A combined note reasons across one X-ray and one MRI of the same knee. */
export type NoteModality = "xray" | "mri" | "combined";

export type ClinicalNoteConfig = "ungrounded" | "rag_no_gate" | "rag_gate";

export type ClinicalNoteFallbackReason =
  | "llm_unavailable"
  | "timeout"
  | "gate_rejected"
  | "parse_failed"
  | "empty_corpus";

export type ClinicalNoteParseMode = "json" | "prose_fallback";

export type ClinicalNotePatientContext = {
  age?: number;
  gender?: string;
  bmi?: number;
  pain_level?: number;
  history?: string;
  symptoms?: string;
};

export type ClinicalNoteRequest = {
  modality: NoteModality;
  grade: number;
  /** Percentage, 0–100. */
  confidence: number;
  /** MRI only; probabilities are percentages. */
  positive_labels?: { name: string; probability: number }[];
  patient_context?: ClinicalNotePatientContext;
  config?: ClinicalNoteConfig;
  /**
   * Combined notes only: one X-ray and one MRI study. The backend recomputes
   * `grade` and `confidence` from these, so the values sent above are ignored.
   */
  studies?: StudyResult[];
};

export type RetrievedPassage = {
  passage_id: string;
  source: string;
  text: string;
  score: number;
  licence: string;
  /** "pmc" for published literature, "internal" for project reference docs. */
  source_type: string;
  url?: string | null;
};

export type NoteSentence = {
  text: string;
  /** 0–1 entailment probability against the best-supporting passage. */
  groundedness: number;
  label: "entailment" | "neutral" | "contradiction";
  /**
   * Derived from the entailment model, not from the marker the model wrote, so
   * it cannot be fabricated.
   */
  supporting_passage_id?: string | null;
  /** The `[n]` the model itself cited, kept for comparison. */
  cited_marker?: number | null;
};

export type TriadScores = {
  /** Null — not zero — for the ungrounded config, where it is undefined. */
  context_relevance: number | null;
  answer_relevance: number;
  groundedness: number;
};

export type ClinicalNote = {
  synthesis: string;
  next_steps: string[];
  sentences: NoteSentence[];
};

export type ClinicalNoteResponse = {
  /**
   * Null when the note was withheld — see `fallback_reason`. Never substituted
   * with deterministic `FeedbackBundle` text: the backend has no static note
   * variant, so a null here always means no model output was served.
   */
  clinical_note: ClinicalNote | null;
  generated_by: string | null;
  /** Pooled mean over every sentence. Reported, but no longer what gates. */
  grounding_score: number | null;
  /**
   * The gate judges the synthesis and the next steps separately, each against
   * its own threshold. Optional so an older backend still type-checks.
   */
  synthesis_groundedness?: number | null;
  /** Null when the note made no recommendations. */
  next_steps_groundedness?: number | null;
  triad: TriadScores | null;
  /** Populated even when the note is withheld. */
  retrieved_passages: RetrievedPassage[];
  gate_attempts: number;
  /** The synthesis threshold. */
  gate_threshold: number;
  next_steps_threshold?: number | null;
  fallback_reason: ClinicalNoteFallbackReason | null;
  parse_mode: ClinicalNoteParseMode | null;
  latency_ms: Record<string, number>;
  config: ClinicalNoteConfig;
};

function postFormWithProgress<T>(
  url: string,
  form: FormData,
  onUploadProgress?: UploadProgressHandler,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);

    xhr.upload.onprogress = (event) => {
      if (!onUploadProgress || !event.lengthComputable) return;
      onUploadProgress(Math.min(100, Math.round((event.loaded / event.total) * 100)));
    };

    xhr.onload = () => {
      let payload: unknown;
      try {
        payload = JSON.parse(xhr.responseText);
      } catch {
        payload = { detail: xhr.statusText || "Request failed" };
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload as T);
        return;
      }
      const detail =
        typeof payload === "object" &&
        payload !== null &&
        "detail" in payload &&
        typeof (payload as { detail: unknown }).detail === "string"
          ? (payload as { detail: string }).detail
          : "Request failed";
      reject(new Error(detail));
    };

    xhr.onerror = () => reject(new Error("Network error while uploading to backbone"));
    xhr.send(form);
  });
}

export async function fetchBackboneHealth(): Promise<BackboneHealth | null> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) return null;
    return res.json() as Promise<BackboneHealth>;
  } catch {
    return null;
  }
}

export async function checkBackboneHealth(): Promise<boolean> {
  const h = await fetchBackboneHealth();
  return h?.status === "healthy";
}

/** Run all available X-ray models when modelNames is omitted or "all". */
export async function predictXray(
  file: File,
  modelNames = "all",
  onUploadProgress?: UploadProgressHandler,
): Promise<XrayPredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_names", modelNames);

  return postFormWithProgress<XrayPredictResponse>(
    `${API_BASE}/api/xray/predict`,
    form,
    onUploadProgress,
  );
}

export async function predictMri(
  file: File,
  threshold?: number,
  onUploadProgress?: UploadProgressHandler,
): Promise<MriPredictResponse> {
  const form = new FormData();
  form.append("file", file);
  if (threshold != null) form.append("threshold", String(threshold));

  return postFormWithProgress<MriPredictResponse>(
    `${API_BASE}/api/mri/predict`,
    form,
    onUploadProgress,
  );
}

/** Run MRI on the pre-loaded Effusion.nii.gz in backbone/ (no client upload). */
export async function predictMriSample(threshold?: number): Promise<MriPredictResponse> {
  const form = new FormData();
  if (threshold != null) form.append("threshold", String(threshold));

  const res = await fetch(`${API_BASE}/api/mri/predict/sample`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : "MRI sample prediction failed");
  }

  return res.json() as Promise<MriPredictResponse>;
}

export function gradcamToDataUrl(base64: string | null | undefined): string | null {
  if (!base64) return null;
  return `data:image/jpeg;base64,${base64}`;
}

export function pngBase64ToDataUrl(base64: string | null | undefined): string | null {
  if (!base64) return null;
  return `data:image/png;base64,${base64}`;
}

/** One modality's result, as sent to the backend orchestrator for fusion. */
export type StudyResult = {
  modality: "xray" | "mri";
  grade: number;
  confidence: number;
  positive_labels?: { name: string; probability: number }[];
};

/**
 * Mirrors `backend/orchestrator.py::FusionResult`.
 *
 * Two caveats travel with every fused grade and must be shown with it: the MRI
 * grade is inferred from cartilage-lesion labels rather than read on the KL
 * scale, and the weights are uncalibrated max-softmax confidences.
 */
export type FusionResult = {
  /** Confidence-weighted mean, rounded half up. */
  grade: number;
  /** Before rounding, so a 2.5 reads as a coin toss rather than a firm 3. */
  weighted_mean: number;
  /** Plain mean of the inputs — never raised for agreement. */
  confidence: number;
  agreement: "concordant" | "discordant";
  grade_gap: number;
  /** Each modality's share of the weighted mean. */
  weights: Partial<Record<"xray" | "mri", number>>;
  studies: StudyResult[];
  method: "confidence_weighted";
};

/** Combine per-modality grades on the backend, the one place fusion happens. */
export async function fuseStudies(
  studies: StudyResult[],
  signal?: AbortSignal,
): Promise<FusionResult> {
  const res = await fetch(`${API_BASE}/api/fusion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ studies }),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : "Fusion failed");
  }

  return res.json() as Promise<FusionResult>;
}

/**
 * Generate a RAG-grounded clinical note for one diagnostic result.
 *
 * A 200 with `clinical_note: null` is a normal outcome, not an error: the note
 * was withheld and `fallback_reason` says why. Only a genuine HTTP failure
 * throws, matching this module's convention.
 */
export async function generateClinicalNote(
  req: ClinicalNoteRequest,
  signal?: AbortSignal,
): Promise<ClinicalNoteResponse> {
  const res = await fetch(`${API_BASE}/api/feedback/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string" ? err.detail : "Clinical note generation failed",
    );
  }

  return res.json() as Promise<ClinicalNoteResponse>;
}
