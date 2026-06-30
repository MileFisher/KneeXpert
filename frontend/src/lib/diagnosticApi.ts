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

/** FLOPs data for a single model. */
export type ModelFlopsEntry = {
  gflops: number;
  params_m: number;
};

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
  model_flops?: Record<string, ModelFlopsEntry>;
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
  model_flops?: Record<string, ModelFlopsEntry>;
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
    xray_models: Record<string, { gflops: number; params_m: number }>;
    mri_models: Record<string, { gflops: number; params_m: number; display_name: string }>;
    total_pipeline_gflops: Record<string, number>;
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
