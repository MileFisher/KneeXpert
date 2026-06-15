import { cn } from "@/lib/utils";

interface ConfidenceGaugeProps {
  value: number | null;
  className?: string;
}

export function ConfidenceGauge({ value, className }: ConfidenceGaugeProps) {
  if (value === null) return <span className="text-xs text-muted-foreground">—</span>;

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative w-0.5 h-6 bg-muted rounded-full overflow-hidden">
        <div
          className="absolute bottom-0 w-full bg-primary rounded-full transition-all duration-500 ease-clinical"
          style={{ height: `${value}%` }}
        />
      </div>
      <span className="text-mono text-xs text-muted-foreground">{value.toFixed(1)}%</span>
    </div>
  );
}
