import type { XrayPredictResponse } from "@/lib/diagnosticApi";
import { gradcamToDataUrl } from "@/lib/diagnosticApi";

export type ReportModelSnapshot = {
  modelId: string;
  displayName: string;
  grade: number;
  confidence: number;
  gradcamDataUrl: string | null;
};

export type ReportDiagnosisAssets = {
  diagnosisSummary: string;
  inputImageDataUrl: string | null;
  ensembleGradcamDataUrl: string | null;
  modelResults: ReportModelSnapshot[];
};

export async function urlToDataUrl(url: string | null): Promise<string | null> {
  if (!url) return null;
  if (url.startsWith("data:")) return url;
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

export function buildDiagnosisSummary(
  grade: number,
  confidence: number,
  findings: string[],
  modelUsed: string,
): string {
  const findingText = findings.length ? findings.join(" ") : "No findings recorded.";
  const confLabel = modelUsed.toLowerCase().includes("ensemble") ? "ensemble" : "model";
  return `KL Grade ${grade} osteoarthritis (${confidence.toFixed(1)}% ${confLabel} confidence). ${findingText} Models: ${modelUsed}.`;
}

export function pngBase64ToDataUrl(base64: string | null | undefined): string | null {
  if (!base64) return null;
  return `data:image/png;base64,${base64}`;
}

export async function buildReportDiagnosisAssets(
  uploadedImageUrl: string | null,
  xrayData: XrayPredictResponse | null,
  grade: number,
  confidence: number,
  findings: string[],
  modelUsed: string,
  mriPreview?: {
    raw_base64?: string | null;
    cleaned_base64?: string | null;
    artifact_map_base64?: string | null;
    gradcam_base64?: string | null;
  } | null,
): Promise<ReportDiagnosisAssets> {
  const fromUpload = await urlToDataUrl(uploadedImageUrl);
  const fromMri = mriPreview?.cleaned_base64
    ? pngBase64ToDataUrl(mriPreview.cleaned_base64)
    : mriPreview?.raw_base64
      ? pngBase64ToDataUrl(mriPreview.raw_base64)
      : null;
  const inputImageDataUrl = fromUpload ?? fromMri;
  const ensembleGradcamDataUrl = xrayData
    ? gradcamToDataUrl(xrayData.gradcam_base64)
    : mriPreview?.gradcam_base64
      ? gradcamToDataUrl(mriPreview.gradcam_base64)
      : mriPreview?.artifact_map_base64
        ? pngBase64ToDataUrl(mriPreview.artifact_map_base64)
        : null;

  const modelResults: ReportModelSnapshot[] = xrayData
    ? Object.entries(xrayData.individual_results).map(([modelId, r]) => ({
        modelId,
        displayName: r.display_name ?? modelId,
        grade: r.grade,
        confidence: r.confidence,
        gradcamDataUrl: gradcamToDataUrl(r.gradcam_base64),
      }))
    : [];

  return {
    diagnosisSummary: buildDiagnosisSummary(grade, confidence, findings, modelUsed),
    inputImageDataUrl,
    ensembleGradcamDataUrl,
    modelResults,
  };
}
