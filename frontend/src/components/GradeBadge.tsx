import { cn } from "@/lib/utils";

interface GradeBadgeProps {
  grade: number | null;
  className?: string;
}

const gradeColors: Record<number, string> = {
  0: "bg-muted text-muted-foreground",
  1: "bg-primary-muted text-primary",
  2: "bg-primary/20 text-primary",
  3: "bg-warning/10 text-warning",
  4: "bg-destructive/10 text-destructive",
};

export function GradeBadge({ grade, className }: GradeBadgeProps) {
  if (grade === null) return <span className="text-xs text-muted-foreground">N/A</span>;
  return (
    <span className={cn("inline-flex items-center justify-center w-7 h-7 rounded-lg text-xs font-semibold", gradeColors[grade] || gradeColors[0], className)}>
      {grade}
    </span>
  );
}
