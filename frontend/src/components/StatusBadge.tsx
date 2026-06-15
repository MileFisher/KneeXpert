import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  status: "pending" | "analyzed" | "confirmed" | "flagged";
  className?: string;
}

const statusConfig = {
  pending: { label: "Pending", className: "bg-muted text-muted-foreground" },
  analyzed: { label: "Analyzed", className: "bg-primary-muted text-primary" },
  confirmed: { label: "Confirmed", className: "bg-success/10 text-success" },
  flagged: { label: "Flagged", className: "bg-warning/10 text-warning" },
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const config = statusConfig[status];
  return (
    <span className={cn("inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium", config.className, className)}>
      {config.label}
    </span>
  );
}
