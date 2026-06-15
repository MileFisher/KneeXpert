import { cn } from "@/lib/utils";

type Props = {
  imageUrl: string | null;
  label: string;
  sublabel?: string;
  className?: string;
  maxHeight?: string;
};

/** Shows a scan or Grad-CAM image at full clarity (no dimmed/blurred base layer). */
export function ScanImageTile({ imageUrl, label, sublabel, className, maxHeight = "280px" }: Props) {
  return (
    <div className={cn("rounded-xl border overflow-hidden bg-muted/20", className)}>
      <div className="relative bg-black/[0.03]" style={{ maxHeight, aspectRatio: "1" }}>
        {imageUrl ? (
          <img src={imageUrl} alt={label} className="w-full h-full object-contain" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
            No preview available
          </div>
        )}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-2 pointer-events-none">
          <p className="text-[10px] font-medium text-white truncate">{label}</p>
          {sublabel && <p className="text-[9px] text-white/80 truncate">{sublabel}</p>}
        </div>
      </div>
    </div>
  );
}
