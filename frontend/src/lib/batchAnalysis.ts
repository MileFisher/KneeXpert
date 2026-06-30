import type { Modality, ModalityReportSnapshot } from "@/data/patients";
import type { CohortInputEntry } from "@/lib/cohortTypes";
import { predictMri, predictMriSample, predictXray, type MriPredictResponse, type XrayPredictResponse } from "@/lib/diagnosticApi";
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
  reliabilityBoost: number;
  agreement: "concordant" | "discordant";
  processedAt: string;
};

export type BatchProgressUpdate = {
  patientId: string;
  progress: number;
  stage: string;
  status: "queued" | "processing" | "completed" | "error";
};

function fuseBatchResults(perModality: BatchModalityResult[]): Pick<
  BatchPatientResult,
  "finalGrade" | "finalConfidence" | "reliabilityBoost" | "agreement"
> {
  const ok = perModality.filter(r => !r.error);
  if (!ok.length) {
    return { finalGrade: 0, finalConfidence: 0, reliabilityBoost: 0, agreement: "discordant" };
  }
  const finalGrade = Math.round(ok.reduce((a, r) => a + r.grade, 0) / ok.length);
  const agreement = ok.every(r => r.grade === ok[0].grade) ? "concordant" : "discordant";
  const avgConf = ok.reduce((a, r) => a + r.confidence, 0) / ok.length;
  const boost = ok.length > 1 && agreement === "concordant" ? 4.2 : ok.length > 1 ? 1.8 : 0;
  return {
    finalGrade,
    finalConfidence: Math.min(99.5, Math.round((avgConf + boost) * 10) / 10),
    reliabilityBoost: boost,
    agreement,
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

  const fused = fuseBatchResults(perModality);
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
  const agreement =
    result.agreement === "concordant"
      ? "Cross-modality concordant."
      : "Cross-modality discordant — review both inputs.";
  return `Fused KL Grade ${result.finalGrade} (${result.finalConfidence.toFixed(1)}% confidence). ${parts.join(" · ")} ${agreement}`;
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
  };
}

function okInputNames(result: BatchPatientResult): string {
  return result.perModality
    .filter(r => !r.error)
    .map(r => r.inputFileName)
    .join(", ");
}
