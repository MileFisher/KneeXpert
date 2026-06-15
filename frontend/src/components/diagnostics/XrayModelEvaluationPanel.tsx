import { useMemo } from "react";
import type { XrayPredictResponse } from "@/lib/diagnosticApi";
import { buildGradcamViewItems } from "@/lib/xrayAnalysis";
import { GradeBadge } from "@/components/GradeBadge";
import { ScanImageTile } from "@/components/diagnostics/ScanImageTile";
import { cn } from "@/lib/utils";

type Props = {
  data: XrayPredictResponse;
  selectedIds: Set<string>;
  showHeatmap: boolean;
  baseImageUrl: string | null;
  inputFileName: string;
};

export function XrayModelEvaluationPanel({
  data,
  selectedIds,
  showHeatmap,
  baseImageUrl,
  inputFileName,
}: Props) {
  const camViews = useMemo(
    () => buildGradcamViewItems(data, selectedIds),
    [data, selectedIds],
  );

  if (camViews.length === 0) {
    return (
      <p className="text-xs text-muted-foreground text-center py-4 border border-dashed rounded-xl">
        Select models in the table above to compare Grad-CAM outputs.
      </p>
    );
  }

  return (
    <div
      className={cn(
        "grid gap-3 mt-3",
        camViews.length === 1 ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2",
      )}
    >
      {camViews.map(view => {
        const showOverlay = showHeatmap && !!view.gradcamUrl;
        const imageUrl = showOverlay ? view.gradcamUrl : baseImageUrl;
        const label = showOverlay ? view.name : inputFileName || view.name;
        const sublabel = showOverlay
          ? `Grade ${view.grade} · ${view.confidence.toFixed(1)}%`
          : `Original input · ${view.name}`;

        return (
          <div key={view.id} className="space-y-1.5">
            <div className="flex items-center justify-between gap-2 px-0.5">
              <span className="text-xs font-medium truncate">{view.name}</span>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <GradeBadge grade={view.grade} />
                <span className="text-mono text-[10px] text-muted-foreground">
                  {view.confidence.toFixed(1)}%
                </span>
              </div>
            </div>
            <ScanImageTile imageUrl={imageUrl} label={label} sublabel={sublabel} />
          </div>
        );
      })}
    </div>
  );
}
