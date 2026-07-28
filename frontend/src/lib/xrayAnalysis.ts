import type { FeedbackBundle, XrayPredictResponse } from "@/lib/diagnosticApi";
import { getModelFlops, gradcamToDataUrl } from "@/lib/diagnosticApi";
import { getFeedbackFindings } from "@/lib/clinicalFeedback";

export type WorkspaceAnalysisResult = {
  grade: number;
  confidence: number;
  findings: string[];
  feedback?: FeedbackBundle;
};

export type ModelPerformanceRow = {
  id: string;
  name: string;
  grade: number;
  confidence: number;
  gradcamUrl: string | null;
  isEnsemble?: boolean;
  /** Highlight final classifier row (MRI DeiT-S — not an ensemble). */
  isPrimary?: boolean;
  /** Override grade column (e.g. pipeline stages). */
  gradeDisplay?: string;
  /** Override confidence column (e.g. slice count). */
  confidenceDisplay?: string;
  /** Multiply-accumulate ops (GMACs) — what model papers usually quote. */
  gmacs?: number;
  /** True floating-point ops (GFLOPs) = 2 x gmacs. */
  gflops?: number;
  /** Parameters in millions. */
  paramsM?: number;
  /** MRI stages only — how many times this model runs per sampled slice. */
  callsPerSlice?: number;
  /** Whether the cost was measured with fvcore or taken from a published figure. */
  costMethod?: "fvcore" | "architecture_estimate";
  /** Set when fvcore's count was rejected as an undercount or is a lower bound. */
  costWarning?: string;
};

/**
 * Tooltip for a model row's compute column.
 *
 * Always states both unit conventions and where the figure came from, so a
 * published estimate is never mistaken for a live measurement.
 */
export function formatModelCostTooltip(row: ModelPerformanceRow): string | undefined {
  if (row.gflops == null) return undefined;
  const parts = [
    `${row.gflops.toFixed(2)} GFLOPs (${row.gmacs?.toFixed(2) ?? "?"} GMACs)`,
    `${row.paramsM?.toFixed(1) ?? "?"}M params`,
  ];
  if (row.callsPerSlice && row.callsPerSlice > 1) {
    parts.push(`runs ${row.callsPerSlice}x per sampled slice`);
  }
  if (row.costMethod) {
    parts.push(
      row.costMethod === "fvcore"
        ? "measured with fvcore"
        : "published architecture estimate",
    );
  }
  const base = parts.join(" · ");
  return row.costWarning ? `${base}\n${row.costWarning}` : base;
}

export type GradcamViewItem = {
  id: string;
  name: string;
  grade: number;
  confidence: number;
  gradcamUrl: string | null;
  isEnsemble?: boolean;
};

export function xrayResponseToResult(data: XrayPredictResponse): WorkspaceAnalysisResult {
  const grade = data.grade;
  return {
    grade,
    confidence: data.confidence,
    findings: getFeedbackFindings("xray", grade, data.feedback),
    feedback: data.feedback,
  };
}

export function buildXrayModelRows(data: XrayPredictResponse): ModelPerformanceRow[] {
  const rows: ModelPerformanceRow[] = Object.entries(data.individual_results).map(([id, r]) => {
    const cost = getModelFlops(data.model_flops, id);
    return {
      id,
      name: r.display_name ?? id,
      grade: r.grade,
      confidence: r.confidence,
      gradcamUrl: gradcamToDataUrl(r.gradcam_base64),
      gmacs: cost?.gmacs,
      gflops: cost?.gflops,
      paramsM: cost?.params_m,
      costMethod: cost?.method,
      costWarning: cost?.measurement_warning,
    };
  });
  rows.push({
    id: "ensemble",
    name: data.ensemble_display_name ?? "Ensemble (probability average)",
    grade: data.grade,
    confidence: data.confidence,
    gradcamUrl: gradcamToDataUrl(data.gradcam_base64),
    isEnsemble: true,
  });
  return rows;
}

export function buildGradcamViewItems(
  data: XrayPredictResponse,
  selectedIds: Set<string>,
): GradcamViewItem[] {
  const items: GradcamViewItem[] = [];
  if (selectedIds.has("ensemble")) {
    items.push({
      id: "ensemble",
      name: data.ensemble_display_name ?? "Ensemble",
      grade: data.grade,
      confidence: data.confidence,
      gradcamUrl: gradcamToDataUrl(data.gradcam_base64),
      isEnsemble: true,
    });
  }
  for (const [id, r] of Object.entries(data.individual_results)) {
    if (!selectedIds.has(id)) continue;
    items.push({
      id,
      name: r.display_name ?? id,
      grade: r.grade,
      confidence: r.confidence,
      gradcamUrl: gradcamToDataUrl(r.gradcam_base64),
    });
  }
  return items;
}

export function defaultSelectedModelIds(data: XrayPredictResponse): Set<string> {
  return new Set(["ensemble", ...Object.keys(data.individual_results)]);
}
