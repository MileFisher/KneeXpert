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
  pipeline_per_sampled_slice?: MriPerSliceCost;
  pipeline_total?: MriStudyCost;
  [modelId: string]: ModelFlopsEntry | MriPerSliceCost | MriStudyCost | string | undefined;
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
      xray_ensemble_all: { gmacs: number; gflops: number; model_count: number };
      mri_per_sampled_slice: MriPerSliceCost;
      mri_per_study_max: MriStudyCost;
    };
  };
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
