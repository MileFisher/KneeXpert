import { useMemo } from "react";
import type { MriPredictResponse } from "@/lib/diagnosticApi";
import {
  buildMriComparisonViews,
  formatVolumeMeta,
  type MriViewMode,
} from "@/lib/mriAnalysis";
import { GradeBadge } from "@/components/GradeBadge";
import { ScanImageTile } from "@/components/diagnostics/ScanImageTile";
import { cn } from "@/lib/utils";

type Props = {
  data: MriPredictResponse;
  selectedIds: Set<string>;
  viewMode: MriViewMode;
  activeSliceIdx: number | null;
  onSliceChange: (idx: number) => void;
  inputFileName: string;
};

export function MriModelEvaluationPanel({
  data,
  selectedIds,
  viewMode,
  activeSliceIdx,
  onSliceChange,
  inputFileName,
}: Props) {
  const views = useMemo(
    () => buildMriComparisonViews(data, selectedIds, viewMode, activeSliceIdx),
    [data, selectedIds, viewMode, activeSliceIdx],
  );

  const gallery = data.slice_gallery ?? [];

  return (
    <div className="mt-3 space-y-3">
      <p className="text-[10px] text-muted-foreground">{formatVolumeMeta(data)}</p>

      {gallery.length > 1 && (
        <div className="flex flex-wrap gap-1.5">
          {gallery.map(s => (
            <button
              key={s.slice_idx}
              type="button"
              onClick={() => onSliceChange(s.slice_idx)}
              className={cn(
                "text-[10px] px-2 py-1 rounded-md border transition-colors",
                activeSliceIdx === s.slice_idx
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-background text-muted-foreground hover:bg-muted",
              )}
            >
              z={s.slice_idx}
            </button>
          ))}
        </div>
      )}

      {views.length === 0 ? (
        <p className="text-xs text-muted-foreground text-center py-4 border border-dashed rounded-xl">
          Select pipeline stages in the table above to compare outputs.
        </p>
      ) : (
        <div
          className={cn(
            "grid gap-3",
            views.length === 1 ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2",
          )}
        >
          {views.map(view => (
            <div key={view.id} className="space-y-1.5">
              <div className="flex items-center justify-between gap-2 px-0.5">
                <span className="text-xs font-medium truncate">{view.name}</span>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  {view.grade != null ? (
                    <>
                      <GradeBadge grade={view.grade} />
                      <span className="text-mono text-[10px] text-muted-foreground">
                        {view.confidence?.toFixed(1)}%
                      </span>
                    </>
                  ) : (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                      Preprocessing
                    </span>
                  )}
                </div>
              </div>
              <ScanImageTile
                imageUrl={view.imageUrl}
                label={view.label}
                sublabel={view.sublabel || inputFileName}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
