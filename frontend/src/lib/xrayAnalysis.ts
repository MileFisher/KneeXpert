import type { XrayPredictResponse } from "@/lib/diagnosticApi";
import { gradcamToDataUrl } from "@/lib/diagnosticApi";
import { getFindingsForGrade } from "@/lib/clinicalFeedback";

export type WorkspaceAnalysisResult = {
  grade: number;
  confidence: number;
  findings: string[];
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
};

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
    findings: getFindingsForGrade("xray", grade),
  };
}

export function buildXrayModelRows(data: XrayPredictResponse): ModelPerformanceRow[] {
  const rows: ModelPerformanceRow[] = Object.entries(data.individual_results).map(([id, r]) => ({
    id,
    name: r.display_name ?? id,
    grade: r.grade,
    confidence: r.confidence,
    gradcamUrl: gradcamToDataUrl(r.gradcam_base64),
  }));
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
