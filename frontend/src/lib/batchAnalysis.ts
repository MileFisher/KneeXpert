import type { Modality, ModalityReportSnapshot } from "@/data/patients";
import type { CohortInputEntry } from "@/lib/cohortTypes";
import {
  fuseStudies,
  predictMri,
  predictMriSample,
  predictXray,
  type FusionResult,
  type MriPredictResponse,
  type XrayPredictResponse,
} from "@/lib/diagnosticApi";
import { mriResponseToResult } from "@/lib/mriAnalysis";
import { buildReportDiagnosisAssets } from "@/lib/reportSnapshot";
import { getFeedbackFindings } from "@/lib/clinicalFeedback";
import { xrayResponseToResult } from "@/lib/xrayAnalysis";
import type { ReportDiagnosisAssets } from "@/lib/reportSnapshot";

export type BatchModalityResult = {
  modality: Modality;
  grade: number;
  confidence: number;
  findings: string[];
  modelUsed: string;
  inputFileName: string;
  view: string;
  region: string;
  xrayData?: XrayPredictResponse;
  mriData?: MriPredictResponse;
  reportAssets?: ReportDiagnosisAssets;
  error?: string;
};

export type BatchPatientResult = {
  patientId: string;
  perModality: BatchModalityResult[];
  finalGrade: number;
  finalConfidence: number;
  agreement: "concordant" | "discordant";
  /** The backend orchestrator's result. Null only when every modality failed. */
  fusion: FusionResult | null;
  processedAt: string;
};

export type BatchProgressUpdate = {
  patientId: string;
  progress: number;
  stage: string;
  status: "queued" | "processing" | "completed" | "error";
};

/**
 * Combine the successful modalities on the backend orchestrator.
 *
 * The rule lives only there: this module used to average the grades itself and
 * add a fixed confidence bonus for agreement, which nothing justified.
 */
async function fuseOnBackend(
  perModality: BatchModalityResult[],
): Promise<
  Pick<BatchPatientResult, "finalGrade" | "finalConfidence" | "agreement" | "fusion">
> {
  const ok = perModality.filter(r => !r.error);
  if (!ok.length) {
    return { finalGrade: 0, finalConfidence: 0, agreement: "discordant", fusion: null };
  }
  const fusion = await fuseStudies(
    ok.map(r => ({
      modality: r.modality,
      grade: r.grade,
      confidence: r.confidence,
      positive_labels: (r.mriData?.multilabel_predictions ?? [])
        .filter(label => label.predicted)
        .map(label => ({ name: label.name, probability: label.probability })),
    })),
  );
  return {
    finalGrade: fusion.grade,
    finalConfidence: fusion.confidence,
    agreement: fusion.agreement,
    fusion,
  };
}

async function analyzeXray(
  file: File,
  region: string,
  view: string,
): Promise<BatchModalityResult> {
  const data = await predictXray(file, "all", pct => {
    void pct;
  });
  const parsed = xrayResponseToResult(data);
  const modelUsed = `Ensemble · ${data.model_count ?? Object.keys(data.individual_results).length} models`;
  const assets = await buildReportDiagnosisAssets(null, data, parsed.grade, parsed.confidence, parsed.findings, modelUsed);
  return {
    modality: "xray",
    grade: parsed.grade,
    confidence: parsed.confidence,
    findings: parsed.findings,
    modelUsed,
    inputFileName: file.name,
    view,
    region,
    xrayData: data,
    reportAssets: assets,
  };
}

async function analyzeMri(
  file: File | null,
  useSample: boolean,
  region: string,
  view: string,
): Promise<BatchModalityResult> {
  const data = useSample ? await predictMriSample() : await predictMri(file!, undefined, pct => { void pct; });
  const parsed = mriResponseToResult(data);
  const modelUsed = "MACS-Net + DeiT-S";
  const assets = await buildReportDiagnosisAssets(null, null, parsed.grade, parsed.confidence, parsed.findings, modelUsed, data.preview);
  return {
    modality: "mri",
    grade: parsed.grade,
    confidence: parsed.confidence,
    findings: parsed.findings,
    modelUsed,
    inputFileName: data.filename,
    view,
    region,
    mriData: data,
    reportAssets: assets,
  };
}

export async function runPatientBatchAnalysis(
  patientId: string,
  modalities: Modality[],
  inputs: CohortInputEntry | undefined,
  files: Partial<Record<Modality, File>> | undefined,
  region: string,
  views: Partial<Record<Modality, string>>,
  options: { mriUseSample?: boolean } = {},
  onStage?: (stage: string) => void,
): Promise<BatchPatientResult> {
  const perModality: BatchModalityResult[] = [];

  for (const mod of modalities) {
    onStage?.(mod === "xray" ? "Running X-ray ensemble…" : "Running MRI pipeline (MACS-Net → DeiT-S)…");
    const view = views[mod] ?? (mod === "xray" ? "AP" : "Axial");
    try {
      if (mod === "xray") {
        const file = files?.xray;
        if (!file) throw new Error("X-ray file missing");
        perModality.push(await analyzeXray(file, region, view));
      } else {
        const file = files?.mri ?? null;
        const useSample = options.mriUseSample && !file;
        if (!file && !useSample) throw new Error("MRI file missing");
        perModality.push(await analyzeMri(file, useSample, region, view));
      }
    } catch (e) {
      perModality.push({
        modality: mod,
        grade: 0,
        confidence: 0,
        findings: getFeedbackFindings(mod, 0),
        modelUsed: mod === "xray" ? "Ensemble" : "MACS-Net + DeiT-S",
        inputFileName: inputs?.[mod]?.fileName ?? "unknown",
        view,
        region,
        error: e instanceof Error ? e.message : "Analysis failed",
      });
    }
  }

  onStage?.("Fusing results on the orchestrator…");
  const fused = await fuseOnBackend(perModality);
  return {
    patientId,
    perModality,
    ...fused,
    processedAt: new Date().toISOString(),
  };
}

export function buildModalitySnapshots(result: BatchPatientResult): ModalityReportSnapshot[] {
  return result.perModality
    .filter(r => !r.error)
    .map(r => ({
      modality: r.modality,
      grade: r.grade,
      confidence: r.confidence,
      findings: r.findings,
      modelUsed: r.modelUsed,
      inputFileName: r.inputFileName,
      view: r.view,
      inputImageDataUrl: r.reportAssets?.inputImageDataUrl,
      ensembleGradcamDataUrl: r.reportAssets?.ensembleGradcamDataUrl,
      modelResults: r.reportAssets?.modelResults,
    }));
}

/**
 * Shown wherever a fused grade is. The MRI grade is inferred from cartilage
 * labels, not read on the KL scale, and the weights are uncalibrated.
 */
export const FUSION_CAVEAT =
  "Weighted by uncalibrated model confidence; the MRI grade is inferred from cartilage-lesion labels, not read on the KL scale.";

export function buildFusedFindings(result: BatchPatientResult): string[] {
  const ok = result.perModality.filter(r => !r.error);
  if (ok.length <= 1) return ok[0]?.findings ?? [];
  return ok.flatMap(r => [
    `[${r.modality === "xray" ? "X-Ray" : "MRI"} KL ${r.grade}]`,
    ...r.findings,
  ]);
}

export function buildFusedDiagnosisSummary(result: BatchPatientResult): string {
  const ok = result.perModality.filter(r => !r.error);
  if (!ok.length) return "Batch analysis failed for all modalities.";
  const parts = ok.map(
    r => `${r.modality === "xray" ? "X-Ray" : "MRI"}: KL ${r.grade} (${r.confidence.toFixed(1)}%) via ${r.modelUsed}`,
  );
  if (ok.length === 1) {
    return `KL Grade ${result.finalGrade} (${result.finalConfidence.toFixed(1)}% confidence). ${parts[0]}.`;
  }
  const agreement =
    result.agreement === "concordant"
      ? "Cross-modality concordant."
      : "Cross-modality discordant — review both inputs.";
  return `Confidence-weighted KL Grade ${result.finalGrade} (mean confidence ${result.finalConfidence.toFixed(1)}%). ${parts.join(" · ")} ${agreement} ${FUSION_CAVEAT}`;
}

export function batchResultToApplyPayload(
  result: BatchPatientResult,
  primaryMod?: Modality,
) {
  const mod = primaryMod ?? result.perModality.find(r => !r.error)?.modality ?? "xray";
  const row = result.perModality.find(r => r.modality === mod && !r.error)
    ?? result.perModality.find(r => !r.error);
  if (!row) return null;
  const assets = row.reportAssets;
  const region = row.region;
  return {
    grade: result.finalGrade,
    aiConfidence: result.finalConfidence,
    findings: buildFusedFindings(result),
    diagnosisSummary: buildFusedDiagnosisSummary(result),
    modality: row.modality,
    view: row.view,
    region,
    inputFileName: okInputNames(result) || row.inputFileName,
    modelUsed: row.modelUsed,
    inputImageDataUrl: assets?.inputImageDataUrl,
    ensembleGradcamDataUrl: assets?.ensembleGradcamDataUrl,
    modelResults: assets?.modelResults,
    modalitySnapshots: buildModalitySnapshots(result),
    fusion: result.fusion ?? undefined,
  };
}

function okInputNames(result: BatchPatientResult): string {
  return result.perModality
    .filter(r => !r.error)
    .map(r => r.inputFileName)
    .join(", ");
}
