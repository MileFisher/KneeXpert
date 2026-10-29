import { afterEach, describe, expect, it, vi } from "vitest";

import type { FusionResult } from "./diagnosticApi";

// Only the network edges and the heavy image-asset builder are replaced; the
// batch flow itself runs for real.
vi.mock("@/lib/diagnosticApi", () => ({
  predictXray: vi.fn(async () => ({ model_count: 8, individual_results: {} })),
  predictMri: vi.fn(async () => ({
    filename: "knee.nii.gz",
    multilabel_predictions: [
      { name: "Effusion", probability: 84.7, predicted: true },
      { name: "Ligament Tear", probability: 12.0, predicted: false },
    ],
  })),
  predictMriSample: vi.fn(),
  fuseStudies: vi.fn(),
}));
vi.mock("@/lib/xrayAnalysis", () => ({
  xrayResponseToResult: () => ({ grade: 1, confidence: 90, findings: ["x"] }),
}));
vi.mock("@/lib/mriAnalysis", () => ({
  mriResponseToResult: () => ({ grade: 4, confidence: 60, findings: ["m"] }),
}));
vi.mock("@/lib/reportSnapshot", () => ({
  buildReportDiagnosisAssets: vi.fn(async () => ({})),
}));

import { fuseStudies } from "@/lib/diagnosticApi";
import {
  FUSION_CAVEAT,
  buildFusedDiagnosisSummary,
  runPatientBatchAnalysis,
} from "./batchAnalysis";

const FUSED: FusionResult = {
  grade: 2,
  weighted_mean: 2.2,
  confidence: 75,
  agreement: "discordant",
  grade_gap: 3,
  weights: { xray: 0.6, mri: 0.4 },
  studies: [],
  method: "confidence_weighted",
};

function run() {
  return runPatientBatchAnalysis(
    "PT-1",
    ["xray", "mri"],
    undefined,
    {
      xray: new File(["x"], "knee.png"),
      mri: new File(["m"], "knee.nii.gz"),
    },
    "Knee",
    {},
  );
}

afterEach(() => {
  vi.mocked(fuseStudies).mockReset();
});

describe("runPatientBatchAnalysis", () => {
  it("takes the final grade from the orchestrator, not a local average", async () => {
    vi.mocked(fuseStudies).mockResolvedValue(FUSED);
    const result = await run();
    // A local round(mean(1, 4)) would give 3; the orchestrator said 2.
    expect(result.finalGrade).toBe(2);
    expect(result.finalConfidence).toBe(75);
    expect(result.fusion).toEqual(FUSED);
  });

  it("sends both studies, with the MRI's positive labels only", async () => {
    vi.mocked(fuseStudies).mockResolvedValue(FUSED);
    await run();
    expect(fuseStudies).toHaveBeenCalledWith([
      { modality: "xray", grade: 1, confidence: 90, positive_labels: [] },
      {
        modality: "mri",
        grade: 4,
        confidence: 60,
        positive_labels: [{ name: "Effusion", probability: 84.7 }],
      },
    ]);
  });

  it("fails loudly when the orchestrator is unreachable", async () => {
    vi.mocked(fuseStudies).mockRejectedValue(new Error("Fusion failed"));
    await expect(run()).rejects.toThrow("Fusion failed");
  });
});

describe("buildFusedDiagnosisSummary", () => {
  it("states the caveat when two studies were combined", async () => {
    vi.mocked(fuseStudies).mockResolvedValue(FUSED);
    const summary = buildFusedDiagnosisSummary(await run());
    expect(summary).toContain("Confidence-weighted");
    expect(summary).toContain(FUSION_CAVEAT);
  });
});
