import { motion } from "framer-motion";
import { Check, AlertTriangle, Loader2 } from "lucide-react";
import type { ScanEntry } from "@/data/patients";

interface MriPipelinePanelProps {
  scan: ScanEntry | undefined;
}

const pipelineStages = [
  {
    id: "pretrain",
    label: "Pre-training",
    description: "Swin-UNet trained on KMAR-50K dataset for deep feature reconstruction",
    detail: "Learning artifact patterns from 50,000 corrupted MRI samples",
  },
  {
    id: "enhance",
    label: "Artifact Removal",
    description: "Raw MRI passed through pre-trained model for artifact removal",
    detail: "Removing motion blur & stripe artifacts while preserving pathology",
  },
  {
    id: "diagnose",
    label: "Diagnosis",
    description: "Cleaned data used for high-precision classification",
    detail: "Enhanced detection of subtle soft-tissue lesions",
  },
];

export function MriPipelinePanel({ scan }: MriPipelinePanelProps) {
  const artifactRemoval = scan?.artifactRemoval;
  const isApplied = artifactRemoval?.applied;

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="border-t bg-muted/10"
    >
      <div className="px-4 py-3">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-5 h-5 rounded bg-primary/10 flex items-center justify-center">
            <svg className="w-3 h-3 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </div>
          <span className="text-xs font-medium">MRI Enhancement Pipeline</span>
          {isApplied && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-success/10 text-success font-medium ml-auto">
              Active
            </span>
          )}
        </div>

        <div className="space-y-2">
          {pipelineStages.map((stage, i) => (
            <div key={stage.id} className="flex items-start gap-3">
              {/* Status indicator */}
              <div className="flex flex-col items-center mt-0.5">
                <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] ${
                  isApplied
                    ? "bg-success/10 text-success"
                    : "bg-muted text-muted-foreground"
                }`}>
                  {isApplied ? <Check className="w-3 h-3" /> : <span>{i + 1}</span>}
                </div>
                {i < pipelineStages.length - 1 && (
                  <div className={`w-px h-6 ${isApplied ? "bg-success/30" : "bg-border"}`} />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-[11px] font-medium">{stage.label}</p>
                <p className="text-[10px] text-muted-foreground leading-relaxed">{stage.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Artifact Removal Stats */}
        {isApplied && artifactRemoval && (
          <div className="mt-3 grid grid-cols-3 gap-2">
            <div className="p-2 rounded-lg bg-background border text-center">
              <p className="text-[10px] text-muted-foreground">Method</p>
              <p className="text-[11px] font-medium">{artifactRemoval.method}</p>
            </div>
            <div className="p-2 rounded-lg bg-background border text-center">
              <p className="text-[10px] text-muted-foreground">Training Data</p>
              <p className="text-[11px] font-medium">{artifactRemoval.dataset}</p>
            </div>
            <div className="p-2 rounded-lg bg-background border text-center">
              <p className="text-[10px] text-muted-foreground">Quality Score</p>
              <p className={`text-[11px] font-medium ${
                artifactRemoval.qualityScore >= 85 ? "text-success" :
                artifactRemoval.qualityScore >= 70 ? "text-warning" : "text-destructive"
              }`}>{artifactRemoval.qualityScore}%</p>
            </div>
          </div>
        )}

        {/* Pre-processing steps */}
        {scan && scan.preprocessing.length > 0 && (
          <div className="mt-3 flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] text-muted-foreground">Pipeline:</span>
            {scan.preprocessing.map((step, i) => (
              <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                {step}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
