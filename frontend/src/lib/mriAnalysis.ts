import type { FeedbackBundle, MriPredictResponse, MriSliceGalleryEntry } from "@/lib/diagnosticApi";
import { gradcamToDataUrl, pngBase64ToDataUrl } from "@/lib/diagnosticApi";
import type { ModelPerformanceRow, WorkspaceAnalysisResult } from "@/lib/xrayAnalysis";

export type MriViewMode = "raw" | "cleaned" | "artifact" | "gradcam";

export type MriComparisonView = {
  id: string;
  name: string;
  grade: number | null;
  confidence: number | null;
  imageUrl: string | null;
  label: string;
  sublabel: string;
};

export function mriResponseToResult(data: MriPredictResponse): WorkspaceAnalysisResult {
  // Prefer backend structured feedback, then fall back to legacy fields
  const findings =
    data.feedback?.key_findings?.length
      ? data.feedback.key_findings
      : data.category_feedback?.length
        ? data.category_feedback
        : data.pathology_findings?.length
          ? data.pathology_findings
          : data.findings;

  return {
    grade: data.grade,
    confidence: data.confidence,
    findings,
    feedback: data.feedback,
  };
}

export function buildMriModelRows(data: MriPredictResponse): ModelPerformanceRow[] {
  const flopsMap = data.model_flops ?? {};
  return [
    {
      id: "macs-net",
      name: "MACS-Net (Swin-UNETR)",
      grade: 0,
      confidence: 100,
      gradcamUrl: null,
      gradeDisplay: "Applied",
      confidenceDisplay: `${data.slices_processed} slices · axis ${data.volume_meta?.slice_axis ?? 2}`,
      gflops: flopsMap["macs_net"]?.gflops,
      paramsM: flopsMap["macs_net"]?.params_m,
    },
    {
      id: "deit-s",
      name: "DeiT-S (2.5D multi-label)",
      grade: data.grade,
      confidence: data.confidence,
      gradcamUrl: gradcamToDataUrl(data.gradcam_base64 ?? data.preview?.gradcam_base64),
      isPrimary: true,
      gflops: flopsMap["deit_small"]?.gflops,
      paramsM: flopsMap["deit_small"]?.params_m,
    },
  ];
}

export function defaultSelectedMriStageIds(): Set<string> {
  return new Set(["macs-net", "deit-s"]);
}

export function getActiveGallerySlice(
  data: MriPredictResponse,
  sliceIdx: number | null,
): MriSliceGalleryEntry | null {
  const gallery = data.slice_gallery ?? [];
  if (!gallery.length) return null;
  if (sliceIdx != null) {
    return gallery.find(s => s.slice_idx === sliceIdx) ?? gallery[0];
  }
  const primary = data.primary_slice_idx ?? data.preview?.center_slice_idx;
  return gallery.find(s => s.slice_idx === primary) ?? gallery[0];
}

export function galleryImageForMode(
  slice: MriSliceGalleryEntry | null,
  preview: MriPredictResponse["preview"] | undefined,
  mode: MriViewMode,
): string | null {
  const fromSlice = (() => {
    if (!slice) return null;
    switch (mode) {
      case "raw":
        return pngBase64ToDataUrl(slice.raw_base64);
      case "cleaned":
        return pngBase64ToDataUrl(slice.cleaned_base64);
      case "artifact":
        return pngBase64ToDataUrl(slice.artifact_map_base64);
      case "gradcam":
        return gradcamToDataUrl(slice.gradcam_base64);
      default:
        return null;
    }
  })();
  if (fromSlice) return fromSlice;
  if (!preview) return null;
  switch (mode) {
    case "raw":
      return pngBase64ToDataUrl(preview.raw_base64);
    case "cleaned":
      return pngBase64ToDataUrl(preview.cleaned_base64);
    case "artifact":
      return pngBase64ToDataUrl(preview.artifact_map_base64);
    case "gradcam":
      return gradcamToDataUrl(preview.gradcam_base64);
    default:
      return null;
  }
}

export function mriViewModeLabel(mode: MriViewMode): string {
  switch (mode) {
    case "raw":
      return "Raw";
    case "cleaned":
      return "MACS cleaned";
    case "artifact":
      return "Artifact map";
    case "gradcam":
      return "DeiT Grad-CAM";
  }
}

export function buildMriComparisonViews(
  data: MriPredictResponse,
  selectedIds: Set<string>,
  viewMode: MriViewMode,
  activeSliceIdx: number | null,
): MriComparisonView[] {
  const slice = getActiveGallerySlice(data, activeSliceIdx);
  const axis = data.volume_meta?.slice_axis ?? slice?.slice_axis ?? 2;
  const zLabel = slice ? `${axis}=${slice.slice_idx}` : "—";
  const views: MriComparisonView[] = [];

  if (selectedIds.has("macs-net")) {
    const url =
      viewMode === "artifact"
        ? galleryImageForMode(slice, data.preview, "artifact")
        : viewMode === "raw"
          ? galleryImageForMode(slice, data.preview, "raw")
          : galleryImageForMode(slice, data.preview, "cleaned");
    views.push({
      id: "macs-net",
      name: "MACS-Net",
      grade: null,
      confidence: null,
      imageUrl: url,
      label: mriViewModeLabel(viewMode === "gradcam" ? "artifact" : viewMode),
      sublabel: `Slice ${zLabel} · ${data.slices_processed} sampled from ${data.volume_meta?.num_slices ?? "?"} planes`,
    });
  }

  if (selectedIds.has("deit-s")) {
    const url =
      viewMode === "gradcam"
        ? galleryImageForMode(slice, data.preview, "gradcam")
        : galleryImageForMode(slice, data.preview, "cleaned");
    views.push({
      id: "deit-s",
      name: "DeiT-S",
      grade: data.grade,
      confidence: data.confidence,
      imageUrl: url,
      label: viewMode === "gradcam" ? "DeiT Grad-CAM" : "2.5D classification input",
      sublabel: slice?.predicted_labels?.length
        ? `Slice ${zLabel}: ${slice.predicted_labels.join(", ")}`
        : `Study KL ${data.grade} · ${data.confidence.toFixed(1)}%`,
    });
  }

  return views;
}

export function mriPreviewToDataUrl(base64: string | null | undefined): string | null {
  return pngBase64ToDataUrl(base64);
}

export function formatVolumeMeta(data: MriPredictResponse): string {
  const m = data.volume_meta;
  if (m?.shape?.length) {
    const orient = m.orientation_axcodes?.length ? m.orientation_axcodes.join("-") : "unknown";
    const planes = m.num_slices ?? m.shape[m.slice_axis] ?? "?";
    return `${m.shape.join("×")} · axis ${m.slice_axis} (${planes} planes) · ${orient}`;
  }
  const shape = data.volume_shape?.length ? data.volume_shape.join("×") : "unknown shape";
  const slices = data.slices_processed ?? "?";
  return `${shape} · ${slices} slices processed`;
}
