import { X, Calendar, Scan, FileText, MessageSquare, ArrowRight } from "lucide-react";
import { Patient } from "@/data/patients";
import { StatusBadge } from "./StatusBadge";
import { GradeBadge } from "./GradeBadge";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

interface PatientDetailPanelProps {
  patient: Patient;
  onClose: () => void;
}

const timelineIcons = {
  scan: Scan,
  diagnosis: FileText,
  note: MessageSquare,
  report: FileText,
};

export function PatientDetailPanel({ patient, onClose }: PatientDetailPanelProps) {
  const navigate = useNavigate();

  return (
    <motion.div
      initial={{ x: 100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 100, opacity: 0 }}
      transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
      className="w-[380px] border-l bg-background h-screen overflow-auto flex-shrink-0"
    >
      <div className="p-5 space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-mono text-xs text-muted-foreground">{patient.id}</p>
            <h2 className="text-lg font-medium mt-0.5">{patient.name}</h2>
            <StatusBadge status={patient.status} className="mt-1.5" />
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-muted transition-colors">
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="card-clinical !p-3">
            <p className="section-header mb-1">Age / Gender</p>
            <p className="text-sm font-medium">{patient.age} / {patient.gender}</p>
          </div>
          <div className="card-clinical !p-3">
            <p className="section-header mb-1">BMI</p>
            <p className="text-sm font-medium">{patient.bmi}</p>
          </div>
          <div className="card-clinical !p-3">
            <p className="section-header mb-1">OA Grade</p>
            <div className="flex items-center gap-2 mt-0.5">
              <GradeBadge grade={patient.grade} />
              <ConfidenceGauge value={patient.aiConfidence} />
            </div>
          </div>
          <div className="card-clinical !p-3">
            <p className="section-header mb-1">Pain Level</p>
            <div className="flex items-center gap-1.5">
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full" style={{ width: `${(patient.painLevel / 10) * 100}%` }} />
              </div>
              <span className="text-mono text-xs">{patient.painLevel}/10</span>
            </div>
          </div>
        </div>

        {/* History */}
        <div>
          <p className="section-header mb-2">Medical History</p>
          <p className="text-sm text-muted-foreground leading-relaxed">{patient.history}</p>
        </div>
        <div>
          <p className="section-header mb-2">Current Symptoms</p>
          <p className="text-sm text-muted-foreground leading-relaxed">{patient.symptoms}</p>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={() => navigate(`/diagnostics?patient=${patient.id}`)}
            className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
          >
            <Scan className="w-4 h-4" />
            Diagnose
          </button>
          <button
            onClick={() => navigate(`/reports?patient=${patient.id}`)}
            className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors"
          >
            <FileText className="w-4 h-4" />
            Report
          </button>
        </div>

        {/* Timeline */}
        <div>
          <p className="section-header mb-3">Patient Timeline</p>
          <div className="space-y-0">
            {patient.timeline.map((entry, i) => {
              const Icon = timelineIcons[entry.type];
              return (
                <div key={i} className="flex gap-3 pb-4 relative">
                  {i < patient.timeline.length - 1 && (
                    <div className="absolute left-[11px] top-7 bottom-0 w-px bg-border" />
                  )}
                  <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Icon className="w-3 h-3 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">{entry.date}</p>
                    <p className="text-sm mt-0.5">{entry.summary}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
