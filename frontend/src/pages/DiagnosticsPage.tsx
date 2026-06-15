import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft, Check, X, Sun, Contrast, Maximize2, Layers, Upload, Image, FileImage,
  Loader2, CheckCircle2, Brain, Sparkles, AlertTriangle, User, Calendar,
  ChevronRight, Search, SlidersHorizontal, Clock, Scan, Type, RotateCw,
  Grid3X3, List, Play, Pause, RefreshCw, Download, Save, Trash2, Move, GripVertical,
  BookOpen, Activity, ShieldCheck, Timer, Users, ArrowRight, Stethoscope, FlaskConical
} from "lucide-react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { type Patient, type Modality } from "@/data/patients";
import { usePatients, type ConfirmDiagnosisPayload } from "@/context/PatientContext";
import { predictXray, predictMri, predictMriSample, fetchBackboneHealth, type XrayPredictResponse, type MriPredictResponse } from "@/lib/diagnosticApi";
import {
  buildGradcamViewItems,
  buildXrayModelRows,
  defaultSelectedModelIds,
  xrayResponseToResult,
  type ModelPerformanceRow,
} from "@/lib/xrayAnalysis";
import {
  buildMriModelRows,
  defaultSelectedMriStageIds,
  formatVolumeMeta,
  galleryImageForMode,
  getActiveGallerySlice,
  mriResponseToResult,
  mriViewModeLabel,
  type MriViewMode,
} from "@/lib/mriAnalysis";
import { MriModelEvaluationPanel } from "@/components/diagnostics/MriModelEvaluationPanel";
import { XrayModelEvaluationPanel } from "@/components/diagnostics/XrayModelEvaluationPanel";
import { ScanImageTile } from "@/components/diagnostics/ScanImageTile";
import { buildReportDiagnosisAssets, buildDiagnosisSummary } from "@/lib/reportSnapshot";
import {
  batchResultToApplyPayload,
  runPatientBatchAnalysis,
  type BatchPatientResult,
} from "@/lib/batchAnalysis";
import type { CohortFilesMap } from "@/lib/cohortFiles";
import type { CohortInputEntry, ModalityUpload } from "@/lib/cohortTypes";
import { gradcamToDataUrl, pngBase64ToDataUrl } from "@/lib/diagnosticApi";
import {
  getFindingsForGrade,
  getGradeNarrative,
  getRecommendationForGrade,
} from "@/lib/clinicalFeedback";
import { GradeBadge } from "@/components/GradeBadge";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { StatusBadge } from "@/components/StatusBadge";
import { DiagnosticsToolbar } from "@/components/diagnostics/DiagnosticsToolbar";
import { MriPipelinePanel } from "@/components/diagnostics/MriPipelinePanel";
import { KonvaImageEditor, type KonvaImageEditorHandle, type EditorTool } from "@/components/diagnostics/KonvaImageEditor";
import { cn } from "@/lib/utils";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription,
} from "@/components/ui/dialog";

// --- Constants ---
const xrayModels = [
  { id: "ensemble", name: "Ensemble (Majority Vote)", description: "ResNet50 + DenseNet201 + VGG-19", accuracy: "95.1%" },
  { id: "densenet", name: "DenseNet201", description: "Detailed classification", accuracy: "94.2%" },
  { id: "vit", name: "ViT-B/16", description: "Global context analysis", accuracy: "92.8%" },
  { id: "resnet", name: "ResNet50", description: "Baseline comparison", accuracy: "89.5%" },
];
const mriModels = [
  { id: "deit-s", name: "DEiT-S", description: "Data-efficient Image Transformer (Small) — Swin-UNet cleaned input", accuracy: "92.4%" },
];
const xrayViews = ["AP", "Lateral"];
const mriViews = ["Sagittal", "Coronal", "Axial"];

type DiagnosticStage = "idle" | "ready" | "uploading" | "preprocessing" | "artifact-removal" | "inference" | "gradcam" | "complete";

type ModalitySession = {
  fileName: string;
  imageUrl: string | null;
  stage: DiagnosticStage;
  stagesCompleted: string[];
  currentStageIndex: number;
  uploadProgress: number;
  feedbackConfirmed: boolean;
  selectedView: string;
};

const emptyModalitySession = (mod: Modality): ModalitySession => ({
  fileName: "",
  imageUrl: null,
  stage: "idle",
  stagesCompleted: [],
  currentStageIndex: 0,
  uploadProgress: 0,
  feedbackConfirmed: false,
  selectedView: mod === "xray" ? xrayViews[0] : mriViews[0],
});

export type { CohortInputEntry, ModalityUpload } from "@/lib/cohortTypes";

const xrayAcceptString = ".dcm,.dicom,.jpg,.jpeg,.png";

// MRI supported input formats (informational — pipeline auto-detects)
const mriSupportedFormats = [
  "DICOM (.dcm)",
  "NIfTI (.nii, .nii.gz)",
  "NRRD (.nrrd, .nhdr)",
  "MetaImage (.mha, .mhd)",
  "Analyze (.img, .hdr)",
  "MINC (.mnc)",
  "PAR/REC (.par, .rec)",
  "Pickle (.pkl, .pck)",
];
const mriAcceptString = ".dcm,.dicom,.nii,.gz,.nii.gz,.nrrd,.nhdr,.mha,.mhd,.img,.hdr,.mnc,.par,.rec,.pkl,.pck,application/gzip,application/x-gzip";

const MRI_FILE_EXTENSIONS = [
  ".dcm", ".dicom", ".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd",
  ".img", ".hdr", ".mnc", ".par", ".rec", ".pkl", ".pck",
] as const;

function isValidModalityFile(file: File, modality: Modality): boolean {
  const name = file.name.toLowerCase();
  if (modality === "xray") {
    return [".dcm", ".dicom", ".jpg", ".jpeg", ".png"].some(ext => name.endsWith(ext));
  }
  return MRI_FILE_EXTENSIONS.some(ext => name.endsWith(ext));
}

function acceptStringForModality(mod: Modality): string {
  return mod === "xray" ? xrayAcceptString : mriAcceptString;
}

function cohortInputKey(patientId: string, modality: Modality): string {
  return `${patientId}:${modality}`;
}

const xrayStages: { id: DiagnosticStage; label: string; duration: number }[] = [
  { id: "uploading", label: "Uploading DICOM file...", duration: 1200 },
  { id: "preprocessing", label: "Pre-processing: CLAHE + Denoise + Normalization", duration: 1800 },
  { id: "inference", label: "Running all X-ray models (8 checkpoints + ensemble)", duration: 2500 },
  { id: "gradcam", label: "Generating Grad-CAM heatmap...", duration: 1200 },
  { id: "complete", label: "Analysis complete", duration: 0 },
];
const mriStages: { id: DiagnosticStage; label: string; duration: number }[] = [
  { id: "uploading", label: "Uploading MRI volume...", duration: 800 },
  { id: "preprocessing", label: "Slice selection along axis 2 (15–85% depth)", duration: 800 },
  { id: "artifact-removal", label: "MACS-Net artifact removal (Swin-UNETR)", duration: 1200 },
  { id: "inference", label: "DeiT-S 2.5D multi-label classification", duration: 1200 },
  { id: "gradcam", label: "Generating artifact maps + DeiT Grad-CAM…", duration: 1000 },
  { id: "complete", label: "Analysis complete", duration: 0 },
];

const mockResults = {
  xray: { grade: 3, confidence: 94.2, findings: getFindingsForGrade("xray", 3) },
  mri: { grade: 2, confidence: 87.6, findings: getFindingsForGrade("mri", 2) },
};

// Per-model performance (mocked) shown after analysis
const modelPerformance = {
  xray: [
    { id: "resnet", name: "ResNet50", grade: 3, confidence: 91.4, gradcamUrl: null, latency: "182 ms", accuracy: "89.5%" },
    { id: "densenet", name: "DenseNet201", grade: 3, confidence: 94.7, gradcamUrl: null, latency: "214 ms", accuracy: "94.2%" },
    { id: "vgg", name: "VGG-19", grade: 3, confidence: 90.1, gradcamUrl: null, latency: "245 ms", accuracy: "88.1%" },
    { id: "ensemble", name: "Ensemble (Majority Vote)", grade: 3, confidence: 94.2, gradcamUrl: null, latency: "641 ms", accuracy: "95.1%" },
  ],
  mri: [
    { id: "deit-s", name: "DEiT-S (on MACS-Net output)", grade: 2, confidence: 87.6, gradcamUrl: null, latency: "298 ms", accuracy: "92.4%" },
  ],
} as const;

// ============================================================
// Patient Selector — unified multi-select (1 or many patients)
// ============================================================
const getPatientModalities = (p: Patient): Modality[] =>
  Array.from(new Set(p.scans.map(s => s.modality))) as Modality[];

// Estimated seconds per scan modality (sum of stage durations / 1000, with overhead)
const estimateSecondsForPatient = (p: Patient): number => {
  const mods = getPatientModalities(p);
  let s = 0;
  if (mods.includes("xray")) s += 7;
  if (mods.includes("mri"))  s += 9;
  if (mods.length > 1)       s += 3;
  return s;
};

function isPatientInputsReady(p: Patient, inputs: CohortInputEntry | undefined): boolean {
  if (!inputs) return false;
  return getPatientModalities(p).every(mod => !!inputs[mod]?.fileName);
}

function PatientSelector({ onConfirm, onOpenHistory }: { onConfirm: (patients: Patient[]) => void; onOpenHistory: () => void }) {
  const { patients } = usePatients();
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [modalityFilter, setModalityFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"name" | "date" | "pain">("date");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [previewPatient, setPreviewPatient] = useState<Patient | null>(null);

  const filtered = useMemo(() => {
    let list = [...patients];
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(p => p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q));
    }
    if (statusFilter !== "all") list = list.filter(p => p.status === statusFilter);
    if (modalityFilter !== "all") {
      list = list.filter(p => getPatientModalities(p).includes(modalityFilter as Modality));
    }
    list.sort((a, b) => {
      if (sortBy === "name") return a.name.localeCompare(b.name);
      if (sortBy === "pain") return b.painLevel - a.painLevel;
      return b.lastVisit.localeCompare(a.lastVisit);
    });
    return list;
  }, [search, statusFilter, modalityFilter, sortBy]);

  const statusOptions = ["all", "pending", "analyzed", "confirmed", "flagged"];
  const urgentCount = patients.filter(p => p.status === "flagged" || p.painLevel >= 7).length;
  const pendingCount = patients.filter(p => p.status === "pending").length;
  const multiModalityCount = patients.filter(p => getPatientModalities(p).length > 1).length;

  const toggle = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectAllFiltered = () => {
    if (filtered.every(p => selected.has(p.id))) {
      const next = new Set(selected);
      filtered.forEach(p => next.delete(p.id));
      setSelected(next);
    } else {
      const next = new Set(selected);
      filtered.forEach(p => next.add(p.id));
      setSelected(next);
    }
  };

  const selectedPatients = patients.filter(p => selected.has(p.id));
  const totalEta = selectedPatients.reduce((s, p) => s + estimateSecondsForPatient(p), 0);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 overflow-auto">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-3 mb-4">
          <div className="min-w-0">
            <h1 className="text-lg sm:text-xl font-semibold tracking-tight">Diagnostic Workspace</h1>
            <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed max-w-2xl">
              Select one patient for single diagnosis, or several for batch analysis. Upload scans before analysis runs.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5 shrink-0">
            <button
              onClick={onOpenHistory}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border bg-background text-xs font-medium hover:bg-muted transition-colors"
            >
              <Clock className="w-3.5 h-3.5" /> History
            </button>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-warning/10 text-warning text-[11px] font-medium">
              <AlertTriangle className="w-3 h-3" /><span className="tabular-nums">{urgentCount}</span> urgent
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-muted text-muted-foreground text-[11px] font-medium">
              <Clock className="w-3 h-3" /><span className="tabular-nums">{pendingCount}</span> pending
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-primary/10 text-primary text-[11px] font-medium">
              <Layers className="w-3 h-3" /><span className="tabular-nums">{multiModalityCount}</span> joint
            </div>
          </div>
        </div>

        {/* Search */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search by name or patient ID..."
            className="w-full pl-9 pr-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/30 transition-shadow"
          />
        </div>

        {/* Filters row */}
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
            {statusOptions.map(s => (
              <button key={s} onClick={() => setStatusFilter(s)}
                className={cn("px-2.5 py-1.5 rounded-md text-xs font-medium transition-all capitalize", statusFilter === s ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground")}
              >{s === "all" ? "All" : s}</button>
            ))}
          </div>
          <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
            {["all", "xray", "mri"].map(m => (
              <button key={m} onClick={() => setModalityFilter(m)}
                className={cn("px-2.5 py-1.5 rounded-md text-xs font-medium transition-all", modalityFilter === m ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground")}
              >{m === "all" ? "All Modalities" : m === "xray" ? "Has X-Ray" : "Has MRI"}</button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            <SlidersHorizontal className="w-3.5 h-3.5 text-muted-foreground" />
            <select value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}
              className="bg-muted rounded-lg px-2.5 py-1.5 text-xs border-0 focus:outline-none cursor-pointer">
              <option value="date">Latest Visit</option>
              <option value="name">Name A-Z</option>
              <option value="pain">Pain Level</option>
            </select>
          </div>
        </div>

        {/* Action bar */}
        <div className="flex items-center justify-between gap-2 mb-2.5 px-3 py-2 rounded-lg border bg-muted/30 sticky top-0 z-10 backdrop-blur-sm">
          <div className="flex items-center gap-2.5 min-w-0 flex-wrap">
            <button onClick={selectAllFiltered} className="text-[11px] text-primary hover:underline font-medium shrink-0">
              {filtered.length > 0 && filtered.every(p => selected.has(p.id)) ? "Deselect all" : "Select all"}
            </button>
            <span className="text-[11px] text-muted-foreground">
              {filtered.length} shown · {selected.size} selected{selected.size > 0 && <> · ~{totalEta}s</>}
            </span>
          </div>
          <button
            onClick={() => onConfirm(selectedPatients)}
            disabled={selected.size === 0}
            className={cn("inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all shrink-0",
              selected.size > 0
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            <ArrowRight className="w-3.5 h-3.5" />
            {selected.size === 1 ? "Open workspace" : `Batch (${selected.size})`}
          </button>
        </div>

        {/* Patient cards — unified multi-select */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2.5">
          {filtered.map(p => {
            const mods = getPatientModalities(p);
            const isSelected = selected.has(p.id);
            const eta = estimateSecondsForPatient(p);
            return (
              <div key={p.id} onClick={() => toggle(p.id)} role="button" tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(p.id); } }}
                className={cn("relative p-3 rounded-lg border bg-card text-left transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring/30",
                  isSelected ? "border-primary ring-1 ring-primary/30 shadow-sm" : "hover:border-border/80 hover:shadow-sm")}>
                <div className="absolute top-2.5 right-2.5 pointer-events-none">
                  <div className={cn("w-4 h-4 rounded border-2 flex items-center justify-center transition-all",
                    isSelected ? "bg-primary border-primary" : "border-muted-foreground/30")}>
                    {isSelected && <Check className="w-2.5 h-2.5 text-primary-foreground" />}
                  </div>
                </div>
                <div className="flex items-center gap-2.5 mb-2 pr-6">
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <User className="w-3.5 h-3.5 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{p.name}</p>
                    <p className="text-[10px] text-muted-foreground font-mono truncate">{p.id} · {p.age}yo</p>
                  </div>
                </div>
                <div className="space-y-1 text-[11px] text-muted-foreground">
                  <div className="flex items-center justify-between gap-1">
                    <div className="flex items-center gap-1 flex-wrap">
                      {mods.includes("xray") && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">X-Ray</span>}
                      {mods.includes("mri") && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">MRI</span>}
                      {mods.length > 1 && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Joint</span>}
                    </div>
                    <StatusBadge status={p.status} />
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span>{p.scans.length} scan{p.scans.length !== 1 ? "s" : ""}</span>
                    <span className="flex items-center gap-0.5 tabular-nums"><Timer className="w-3 h-3" />~{eta}s</span>
                  </div>
                  <p className="text-[10px] line-clamp-2 leading-snug">{p.symptoms}</p>
                </div>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setPreviewPatient(p); }}
                  className="mt-2 w-full inline-flex items-center justify-center gap-1 px-2 py-1 rounded-md border bg-background text-[10px] font-medium text-foreground hover:bg-muted transition-colors"
                >
                  <Scan className="w-3 h-3" /> Inputs & history
                </button>
              </div>
            );
          })}
        </div>

        {filtered.length === 0 && (
          <div className="py-16 flex flex-col items-center gap-3 text-muted-foreground">
            <Search className="w-8 h-8" />
            <p className="text-sm">No patients match your filters</p>
            <button onClick={() => { setSearch(""); setStatusFilter("all"); setModalityFilter("all"); }} className="text-xs text-primary hover:underline">Clear filters</button>
          </div>
        )}
      </div>
      <PatientPreviewDialog patient={previewPatient} onClose={() => setPreviewPatient(null)} />
    </motion.div>
  );
}

// ============================================================
// Patient Preview Dialog — inputs (scans) + prior diagnostic history
// ============================================================
function PatientPreviewDialog({ patient, onClose }: { patient: Patient | null; onClose: () => void }) {
  return (
    <Dialog open={!!patient} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        {patient && (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <User className="w-4 h-4 text-primary" /> {patient.name}
                <span className="text-[10px] font-mono text-muted-foreground">{patient.id}</span>
              </DialogTitle>
              <DialogDescription className="text-xs">
                {patient.age}yo · {patient.gender} · BMI {patient.bmi} · Pain {patient.painLevel}/10
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-5 mt-2">
              {/* Clinical context */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="p-3 rounded-lg border bg-muted/30">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">History</p>
                  <p className="text-xs leading-relaxed">{patient.history}</p>
                </div>
                <div className="p-3 rounded-lg border bg-muted/30">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Symptoms</p>
                  <p className="text-xs leading-relaxed">{patient.symptoms}</p>
                </div>
              </div>

              {/* Inputs (scans) */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                    <Scan className="w-3.5 h-3.5" /> Input scans ({patient.scans.length})
                  </h3>
                </div>
                <div className="rounded-xl border divide-y bg-card">
                  {patient.scans.map((s) => (
                    <div key={s.id} className="p-3 text-xs">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">
                            {s.modality === "xray" ? "X-Ray" : "MRI"}
                          </span>
                          <span className="font-medium truncate">{s.region}{s.view ? ` · ${s.view}` : ""}</span>
                        </div>
                        <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                          <Calendar className="w-3 h-3" />{s.date}
                        </span>
                      </div>
                      <p className="text-[11px] text-muted-foreground">
                        Pipeline: {s.preprocessing.length ? s.preprocessing.join(" → ") : "—"}
                      </p>
                      <div className="flex items-center justify-between mt-1.5">
                        <p className="text-[11px] text-muted-foreground truncate">Model: {s.modelUsed}</p>
                        <div className="flex items-center gap-2">
                          {s.grade != null
                            ? <><GradeBadge grade={s.grade} /><span className="text-[11px] font-semibold tabular-nums">{s.aiConfidence?.toFixed(1)}%</span></>
                            : <span className="text-[10px] text-muted-foreground italic">Not analyzed</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Prior diagnostic history (timeline) */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5 mb-2">
                  <Clock className="w-3.5 h-3.5" /> Prior diagnostic history
                </h3>
                {patient.timeline.length === 0 ? (
                  <p className="text-xs text-muted-foreground py-2">No history yet.</p>
                ) : (
                  <ol className="relative border-l ml-2 pl-4 space-y-3">
                    {patient.timeline.map((t, i) => (
                      <li key={i} className="text-xs">
                        <span className="absolute -left-[5px] w-2 h-2 rounded-full bg-primary mt-1.5" />
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                          <Calendar className="w-3 h-3" />{t.date}
                          <span className="uppercase tracking-wider">{t.type}</span>
                          {t.grade != null && <GradeBadge grade={t.grade} />}
                          {t.confidence != null && <span className="tabular-nums">{t.confidence.toFixed(1)}%</span>}
                        </div>
                        <p className="mt-0.5 leading-relaxed">{t.summary}</p>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// Confirm input dialog — review upload before running analysis
// ============================================================
function ConfirmInputDialog({
  open,
  onClose,
  onConfirm,
  modality,
  view,
  fileName,
  previewUrl,
  patientName,
  uploads,
  mriServerSample = false,
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  modality: Modality;
  view: string;
  fileName: string;
  previewUrl: string | null;
  patientName: string;
  uploads?: { modality: Modality; view: string; fileName: string; previewUrl: string | null }[];
  mriServerSample?: boolean;
}) {
  const items = uploads ?? [{ modality, view, fileName, previewUrl }];
  const isJoint = items.length > 1;
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="max-w-md max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Confirm input before analysis</DialogTitle>
          <DialogDescription className="text-xs">
            Review the uploaded scan{isJoint ? "s" : ""} for <span className="font-medium text-foreground">{patientName}</span>. Analysis will not start until you confirm.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 mt-2">
          {items.map(item => {
            const isXray = item.modality === "xray";
            return (
              <div key={item.modality} className="space-y-3 p-3 rounded-xl border bg-muted/20">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[10px] uppercase px-2 py-0.5 rounded bg-muted font-medium">{isXray ? "X-Ray" : "MRI"}</span>
                  <span className="text-xs text-muted-foreground">View: <span className="font-medium text-foreground">{item.view}</span></span>
                </div>
                {item.previewUrl ? (
                  <div className="rounded-xl border overflow-hidden bg-foreground/[0.02] aspect-video flex items-center justify-center">
                    <img src={item.previewUrl} alt="Input preview" className="max-h-36 w-full object-contain" />
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed p-4 flex flex-col items-center gap-2 text-muted-foreground">
                    <FileImage className="w-6 h-6" />
                    <p className="text-[10px]">
                      {isXray ? "Preview unavailable for this file type" : mriServerSample ? "Server sample — no client upload" : "Preview unavailable for this file type"}
                    </p>
                  </div>
                )}
                <div className="text-xs space-y-1">
                  <p><span className="text-muted-foreground">File:</span> <span className="font-medium">{item.fileName}</span></p>
                  <p className="text-muted-foreground text-[11px]">
                    Pipeline: {isXray
                      ? "CLAHE + Denoise → Ensemble inference → Grad-CAM"
                      : "2.5D slices → MACS-Net → DeiT-S multi-label"}
                  </p>
                  {!isXray && mriServerSample && item.modality === "mri" && (
                    <p className="text-[11px] text-primary">Pre-loaded on backbone — upload skipped.</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
            Back
          </button>
          <button
            onClick={onConfirm}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
          >
            <Check className="w-4 h-4" />Confirm &amp; run analysis
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// Shared clinical helpers — medical references
// ============================================================
const medicalReferences = [
  {
    citation: "Kellgren JH, Lawrence JS. Radiological assessment of osteo-arthrosis. Ann Rheum Dis. 1957;16(4):494–502.",
    note: "Original Kellgren–Lawrence grading scale (KL 0–4) used by this classifier.",
  },
  {
    citation: "Altman RD, Gold GE. Atlas of individual radiographic features in osteoarthritis, revised. Osteoarthritis Cartilage. 2007;15(Suppl A):A1–A56.",
    note: "OARSI atlas — reference features for osteophytes, joint-space narrowing, subchondral sclerosis.",
  },
  {
    citation: "Hunter DJ, Guermazi A, et al. Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS. Osteoarthritis Cartilage. 2011;19(8):990–1002.",
    note: "MOAKS framework for MRI scoring of cartilage, BMLs, menisci.",
  },
  {
    citation: "Tiulpin A, et al. Automatic knee osteoarthritis diagnosis from plain radiographs: a deep learning–based approach. Sci Rep. 2018;8:1727.",
    note: "Validation baseline for CNN ensembles on plain-film KL grading.",
  },
  {
    citation: "Bannur S, et al. Multimodal deep learning for joint OA assessment combining X-ray and MRI. Med Image Anal. 2023;85:102749.",
    note: "Evidence that joint X-ray + MRI fusion improves grading reliability over single-modality models.",
  },
];

interface ModalityResult { modality: Modality; grade: number; confidence: number; }
interface JointAnalysis {
  perModality: ModalityResult[];
  finalGrade: number;
  finalConfidence: number;
  reliabilityBoost: number;
  agreement: "concordant" | "discordant";
}

function computeJointAnalysis(patient: Patient, batchResult?: BatchPatientResult): JointAnalysis {
  if (batchResult) {
    const ok = batchResult.perModality.filter(r => !r.error);
    return {
      perModality: ok.map(r => ({
        modality: r.modality,
        grade: r.grade,
        confidence: r.confidence,
      })),
      finalGrade: batchResult.finalGrade,
      finalConfidence: batchResult.finalConfidence,
      reliabilityBoost: batchResult.reliabilityBoost,
      agreement: batchResult.agreement,
    };
  }

  if (patient.report?.modalitySnapshots?.length) {
    const snaps = patient.report.modalitySnapshots;
    const perModality: ModalityResult[] = snaps.map(s => ({
      modality: s.modality,
      grade: s.grade,
      confidence: Math.round(s.confidence * 10) / 10,
    }));
    const finalGrade = patient.report.finalGrade;
    const agreement = perModality.every(r => r.grade === perModality[0]?.grade) ? "concordant" : "discordant";
    const avgConf = perModality.reduce((a, r) => a + r.confidence, 0) / perModality.length;
    const boost = perModality.length > 1 && agreement === "concordant" ? 4.2 : perModality.length > 1 ? 1.8 : 0;
    return {
      perModality,
      finalGrade,
      finalConfidence: patient.report.aiConfidence,
      reliabilityBoost: boost,
      agreement: perModality.length > 1 ? agreement : "concordant",
    };
  }

  const mods = Array.from(new Set([
    ...patient.scans.map(s => s.modality),
    ...(patient.report ? [patient.report.modality] : []),
  ])) as Modality[];

  const perModality: ModalityResult[] = mods.map(m => {
    if (
      patient.report?.modality === m &&
      patient.grade != null &&
      patient.aiConfidence != null &&
      patient.status !== "pending"
    ) {
      return {
        modality: m,
        grade: patient.grade,
        confidence: Math.round(patient.aiConfidence * 10) / 10,
      };
    }
    const scans = patient.scans.filter(s => s.modality === m && s.grade != null);
    if (!scans.length) {
      return { modality: m, grade: 0, confidence: 0 };
    }
    const grade = Math.round(scans.reduce((a, s) => a + (s.grade ?? 0), 0) / scans.length);
    const confidence = scans.reduce((a, s) => a + (s.aiConfidence ?? 0), 0) / scans.length;
    return { modality: m, grade, confidence: Math.round(confidence * 10) / 10 };
  }).filter(r => r.confidence > 0 || r.grade > 0);

  if (!perModality.length) {
    return { perModality: [], finalGrade: 0, finalConfidence: 0, reliabilityBoost: 0, agreement: "discordant" };
  }

  const finalGrade = Math.round(perModality.reduce((a, r) => a + r.grade, 0) / perModality.length);
  const agreement = perModality.every(r => r.grade === perModality[0].grade) ? "concordant" : "discordant";
  const avgConf = perModality.reduce((a, r) => a + r.confidence, 0) / perModality.length;
  const boost = perModality.length > 1 && agreement === "concordant" ? 4.2 : perModality.length > 1 ? 1.8 : 0;
  return {
    perModality,
    finalGrade,
    finalConfidence: Math.min(99.5, Math.round((avgConf + boost) * 10) / 10),
    reliabilityBoost: boost,
    agreement,
  };
}

// ============================================================
// Clinical Interpretation + References (reused in workspace + overview)
// ============================================================
function ClinicalInterpretation({
  patient,
  analysis,
  batchResult,
  compact = false,
}: {
  patient: Patient;
  analysis: JointAnalysis;
  batchResult?: BatchPatientResult;
  compact?: boolean;
}) {
  const findingsForMod = (mod: Modality, grade: number) => {
    const batchRow = batchResult?.perModality.find(r => r.modality === mod && !r.error);
    if (batchRow?.findings.length) return batchRow.findings;
    const snap = patient.report?.modalitySnapshots?.find(s => s.modality === mod);
    if (snap?.findings.length) return snap.findings;
    if (patient.report?.modality === mod && patient.report.findings.length) return patient.report.findings;
    return getFindingsForGrade(mod, grade);
  };
  return (
    <div className="space-y-3">
      <div className="p-4 rounded-xl border bg-card">
        <div className="flex items-center gap-2 mb-3">
          <Stethoscope className="w-4 h-4 text-primary" />
          <p className="text-sm font-medium">Clinical Interpretation</p>
          {analysis.perModality.length > 1 && (
            <span className={cn("text-[10px] px-1.5 py-0.5 rounded font-medium",
              analysis.agreement === "concordant" ? "bg-success/10 text-success" : "bg-warning/10 text-warning")}>
              {analysis.agreement === "concordant" ? "Cross-modality concordant" : "Cross-modality discordant"}
            </span>
          )}
          {analysis.reliabilityBoost > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
              +{analysis.reliabilityBoost}% reliability
            </span>
          )}
        </div>
        <p className="text-sm leading-relaxed text-foreground/90 mb-3">
          The AI ensemble classified this {patient.age}-year-old {patient.gender.toLowerCase()} patient
          (BMI {patient.bmi}) as <span className="font-medium">Kellgren–Lawrence Grade {analysis.finalGrade} osteoarthritis</span> with
          a fused confidence of {analysis.finalConfidence}%. {getGradeNarrative(analysis.finalGrade)}
          {analysis.perModality.length > 1 && " Multi-modality fusion of plain radiograph and MRI inputs strengthens the structural assessment by combining osseous evaluation from X-ray with soft-tissue (cartilage, meniscus, synovium) evaluation from MRI."}
        </p>
        <div className="space-y-3">
          {analysis.perModality.map(r => {
            const findings = findingsForMod(r.modality, r.grade);
            return (
              <div key={r.modality} className="p-3 rounded-lg bg-muted/30 border">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground">
                    {r.modality === "xray" ? "Plain Radiograph (X-Ray)" : "MRI"} · KL Grade {r.grade}
                  </span>
                  <GradeBadge grade={r.grade} />
                  <span className="text-[10px] text-muted-foreground">{r.confidence}% confidence</span>
                </div>
                <ul className="space-y-1">
                  {findings.map((f, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground leading-relaxed">
                      <span className="w-1 h-1 rounded-full bg-primary mt-1.5 flex-shrink-0" />{f}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
        <div className="mt-3 p-3 rounded-lg bg-primary/5 border border-primary/20">
          <p className="text-[10px] uppercase tracking-wider font-semibold text-primary mb-1">Recommended next steps</p>
          <p className="text-xs text-foreground/90 leading-relaxed">
            {getRecommendationForGrade(analysis.finalGrade)}
          </p>
        </div>
      </div>

      {!compact && (
        <div className="p-4 rounded-xl border bg-card">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-4 h-4 text-primary" />
            <p className="text-sm font-medium">Medical References</p>
            <span className="text-[10px] text-muted-foreground">Evidence base for this classification</span>
          </div>
          <ol className="space-y-2.5 list-decimal list-inside">
            {medicalReferences.map((r, i) => (
              <li key={i} className="text-xs leading-relaxed">
                <span className="text-foreground/90">{r.citation}</span>
                <p className="text-[11px] text-muted-foreground mt-0.5 ml-4">{r.note}</p>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

// ============================================================
// Batch input screen — upload scans for each patient before analysis
// ============================================================
function BatchInputScreen({
  patients,
  onCancel,
  onContinue,
}: {
  patients: Patient[];
  onCancel: () => void;
  onContinue: (inputs: Map<string, CohortInputEntry>, files: CohortFilesMap) => void;
}) {
  const fileRefs = useRef<Map<string, HTMLInputElement>>(new Map());
  const filesMapRef = useRef<CohortFilesMap>(new Map());
  const [inputMap, setInputMap] = useState<Map<string, CohortInputEntry>>(() => new Map());
  const [previewPatient, setPreviewPatient] = useState<Patient | null>(null);

  const handleFile = (patientId: string, modality: Modality, file: File) => {
    if (!isValidModalityFile(file, modality)) {
      toast.error(`Invalid file type for ${modality === "xray" ? "X-Ray" : "MRI"}. Check supported formats.`);
      return;
    }
    const previewUrl = file.type.startsWith("image/") ? URL.createObjectURL(file) : null;
    setInputMap(prev => {
      const next = new Map(prev);
      const existing = next.get(patientId) ?? {};
      if (existing[modality]?.previewUrl) URL.revokeObjectURL(existing[modality]!.previewUrl!);
      next.set(patientId, { ...existing, [modality]: { fileName: file.name, previewUrl } });
      return next;
    });
    const fileEntry = filesMapRef.current.get(patientId) ?? {};
    filesMapRef.current.set(patientId, { ...fileEntry, [modality]: file });
  };

  const clearInput = (patientId: string, modality: Modality) => {
    setInputMap(prev => {
      const next = new Map(prev);
      const existing = next.get(patientId);
      if (existing?.[modality]?.previewUrl) URL.revokeObjectURL(existing[modality]!.previewUrl!);
      if (existing) {
        const updated = { ...existing };
        delete updated[modality];
        if (Object.keys(updated).length === 0) next.delete(patientId);
        else next.set(patientId, updated);
      }
      return next;
    });
    const fileEntry = filesMapRef.current.get(patientId);
    if (fileEntry) {
      const updated = { ...fileEntry };
      delete updated[modality];
      if (Object.keys(updated).length === 0) filesMapRef.current.delete(patientId);
      else filesMapRef.current.set(patientId, updated);
    }
    const ref = fileRefs.current.get(cohortInputKey(patientId, modality));
    if (ref) ref.value = "";
  };

  const readyCount = patients.filter(p => isPatientInputsReady(p, inputMap.get(p.id))).length;
  const allReady = readyCount === patients.length;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 overflow-auto">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
        <div className="flex items-center gap-2.5 mb-3">
          <button onClick={onCancel} className="p-1.5 rounded-lg hover:bg-muted transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-lg font-semibold">Upload Batch Inputs</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Upload each required scan (X-Ray and/or MRI) per patient.</p>
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 mb-3 px-3 py-2 rounded-lg border bg-muted/30 sticky top-0 z-10 backdrop-blur-sm">
          <span className="text-[11px] text-muted-foreground">
            <span className="font-medium text-foreground tabular-nums">{readyCount}</span> / {patients.length} ready
          </span>
          <div className="flex-1 max-w-[200px] h-1.5 bg-muted rounded-full overflow-hidden mx-2 hidden sm:block">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: patients.length ? `${(readyCount / patients.length) * 100}%` : "0%" }}
            />
          </div>
          <span className="text-[11px] font-medium text-primary">{allReady ? "All uploaded" : "Complete all modalities"}</span>
        </div>

        <div className={cn("gap-2.5 mb-4", patients.length > 1 ? "grid grid-cols-1 lg:grid-cols-2" : "space-y-2.5")}>
          {patients.map(p => {
            const inputs = inputMap.get(p.id);
            const mods = getPatientModalities(p);
            const patientReady = isPatientInputsReady(p, inputs);
            return (
              <div key={p.id} className={cn("p-3 rounded-lg border bg-card transition-colors", patientReady ? "border-success/30" : "border-border")}>
                <div className="flex items-start gap-2.5">
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <User className="w-3.5 h-3.5 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{p.name}</p>
                    <p className="text-[10px] text-muted-foreground font-mono">{p.id}</p>
                    <div className="flex items-center gap-1 mt-0.5 flex-wrap">
                      {mods.includes("xray") && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">X-Ray</span>}
                      {mods.includes("mri") && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">MRI</span>}
                      {mods.length > 1 && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Joint</span>}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setPreviewPatient(p)}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md border bg-background text-[10px] font-medium hover:bg-muted transition-colors shrink-0"
                  >
                    <Scan className="w-3 h-3" /> History
                  </button>
                </div>

                <div className={cn("mt-2.5 gap-1.5", mods.length > 1 ? "grid sm:grid-cols-2" : "space-y-1.5")}>
                  {mods.map(mod => {
                    const input = inputs?.[mod];
                    const refKey = cohortInputKey(p.id, mod);
                    return (
                      <div key={mod} className="rounded-lg border bg-muted/20 p-2.5">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-[10px] uppercase font-semibold tracking-wider text-muted-foreground">
                            {mod === "xray" ? "X-Ray input" : "MRI input"}
                          </span>
                          {input && <CheckCircle2 className="w-3.5 h-3.5 text-success" />}
                        </div>
                        <input
                          ref={el => { if (el) fileRefs.current.set(refKey, el); }}
                          type="file"
                          accept={acceptStringForModality(mod)}
                          className="hidden"
                          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(p.id, mod, f); e.target.value = ""; }}
                        />
                        {input ? (
                          <div className="flex items-center gap-3">
                            {input.previewUrl ? (
                              <img src={input.previewUrl} alt="" className="w-12 h-12 rounded-lg object-cover border flex-shrink-0" />
                            ) : (
                              <div className="w-12 h-12 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                                <FileImage className="w-5 h-5 text-muted-foreground" />
                              </div>
                            )}
                            <div className="min-w-0 flex-1">
                              <p className="text-xs font-medium truncate">{input.fileName}</p>
                              <p className="text-[10px] text-success mt-0.5">Ready</p>
                            </div>
                            <button type="button" onClick={() => fileRefs.current.get(refKey)?.click()} className="text-[11px] text-primary hover:underline font-medium">Replace</button>
                            <button type="button" onClick={() => clearInput(p.id, mod)} className="text-[11px] text-destructive hover:underline font-medium">Remove</button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => fileRefs.current.get(refKey)?.click()}
                            className="w-full h-12 rounded-lg border-2 border-dashed border-border hover:border-primary/40 hover:bg-primary/5 flex flex-col items-center justify-center gap-0.5 transition-all"
                          >
                            <Upload className="w-4 h-4 text-muted-foreground" />
                            <span className="text-[11px] font-medium">Upload {mod === "xray" ? "X-Ray" : "MRI"} scan</span>
                            <span className="text-[9px] text-muted-foreground">
                              {mod === "xray" ? "DICOM, JPEG, PNG" : "DICOM, NIfTI (.nii.gz), NRRD, .pkl, .pck, etc."}
                            </span>
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button onClick={onCancel} className="px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
            Back to selection
          </button>
          <button
            onClick={() => onContinue(new Map(inputMap), new Map(filesMapRef.current))}
            disabled={!allReady}
            className={cn(
              "inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-colors",
              allReady ? "bg-primary text-primary-foreground hover:bg-primary/90" : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            Continue to review<ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
      <PatientPreviewDialog patient={previewPatient} onClose={() => setPreviewPatient(null)} />
    </motion.div>
  );
}

// ============================================================
// Confirmation screen — review before starting diagnosis
// ============================================================
function ConfirmationScreen({
  patients,
  cohortInputs,
  onCancel,
  onStart,
}: {
  patients: Patient[];
  cohortInputs: Map<string, CohortInputEntry>;
  onCancel: () => void;
  onStart: () => void;
}) {
  const [previewPatient, setPreviewPatient] = useState<Patient | null>(null);
  const totalEta = patients.reduce((s, p) => s + estimateSecondsForPatient(p), 0);
  const multiModalityPatients = patients.filter(p => getPatientModalities(p).length > 1);
  const totalInputCount = patients.reduce((n, p) => {
    const entry = cohortInputs.get(p.id);
    if (!entry) return n;
    return n + getPatientModalities(p).filter(m => entry[m]?.fileName).length;
  }, 0);
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 overflow-auto">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
        <div className="flex items-center gap-2.5 mb-3">
          <button onClick={onCancel} className="p-1.5 rounded-lg hover:bg-muted transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-lg font-semibold">Confirm Batch Diagnosis</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Review uploads, then start AI analysis.</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 mb-3 p-2.5 rounded-lg border bg-card">
          <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-muted/50 text-xs">
            <Users className="w-3.5 h-3.5 text-primary" />
            <span className="text-muted-foreground">Patients</span>
            <span className="font-semibold tabular-nums">{patients.length}</span>
          </div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-muted/50 text-xs">
            <Upload className="w-3.5 h-3.5 text-primary" />
            <span className="text-muted-foreground">Inputs</span>
            <span className="font-semibold tabular-nums">{totalInputCount}</span>
          </div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-muted/50 text-xs">
            <Timer className="w-3.5 h-3.5 text-primary" />
            <span className="text-muted-foreground">Est.</span>
            <span className="font-semibold tabular-nums">~{totalEta}s</span>
          </div>
        </div>

        {multiModalityPatients.length > 0 && (
          <div className="p-2.5 rounded-lg border border-primary/20 bg-primary/5 mb-3 flex items-start gap-2">
            <ShieldCheck className="w-3.5 h-3.5 text-primary flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-foreground/90 leading-relaxed">
              <span className="font-medium">{multiModalityPatients.length} patient{multiModalityPatients.length !== 1 ? "s" : ""}</span> with X-ray + MRI — joint analysis fuses osseous and soft-tissue signals for a more reliable grade.
            </p>
          </div>
        )}

        <div className="border rounded-lg divide-y mb-4 overflow-hidden max-h-[min(52vh,520px)] overflow-y-auto">
          {patients.map(p => {
            const mods = getPatientModalities(p);
            const input = cohortInputs.get(p.id);
            return (
              <div key={p.id} className="px-3 py-2 grid grid-cols-[auto_1fr_auto] sm:grid-cols-[auto_1fr_auto_auto_auto] items-center gap-x-2.5 gap-y-1">
                <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <User className="w-3.5 h-3.5 text-primary" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{p.name}</p>
                  <p className="text-[10px] text-muted-foreground font-mono truncate">{p.id}</p>
                  {input && (
                    <p className="text-[10px] text-muted-foreground truncate mt-0.5">
                      {mods.filter(mod => input[mod]).map(mod => `${mod === "xray" ? "X-Ray" : "MRI"}: ${input[mod]!.fileName}`).join(" · ")}
                    </p>
                  )}
                </div>
                <div className="hidden sm:flex items-center gap-1 flex-shrink-0">
                  {mods.includes("xray") && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">X-Ray</span>}
                  {mods.includes("mri") && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-muted font-medium">MRI</span>}
                  {mods.length > 1 && <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Joint</span>}
                </div>
                <button
                  type="button"
                  onClick={() => setPreviewPatient(p)}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-md border text-[10px] font-medium hover:bg-muted transition-colors justify-self-end sm:justify-self-auto"
                >
                  <Scan className="w-3 h-3" />
                </button>
                <span className="text-[10px] text-muted-foreground flex items-center gap-0.5 tabular-nums justify-self-end col-span-1 sm:col-auto"><Timer className="w-3 h-3" />~{estimateSecondsForPatient(p)}s</span>
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button onClick={onCancel} className="px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
            Back to uploads
          </button>
          <button onClick={onStart} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
            <Play className="w-4 h-4" />Start Batch Analysis
          </button>
        </div>
      </div>
      <PatientPreviewDialog patient={previewPatient} onClose={() => setPreviewPatient(null)} />
    </motion.div>
  );
}

// ============================================================
// Processing screen — live progress + ETA countdown
// ============================================================
interface PatientProgress { patientId: string; progress: number; stage: string; status: "queued" | "processing" | "completed" | "error"; }

function BatchModalityDetail({ row }: { row: import("@/lib/batchAnalysis").BatchModalityResult }) {
  if (row.error) {
    return (
      <div className="p-3 text-xs border-t bg-destructive/5">
        <p className="font-medium text-destructive">{row.modality === "xray" ? "X-Ray" : "MRI"} analysis failed</p>
        <p className="text-[11px] text-muted-foreground mt-1">{row.error}</p>
      </div>
    );
  }

  const previewUrl = row.modality === "xray"
    ? (row.xrayData ? gradcamToDataUrl(row.xrayData.gradcam_base64) : row.reportAssets?.ensembleGradcamDataUrl)
    : (row.mriData?.preview?.gradcam_base64
      ? gradcamToDataUrl(row.mriData.preview.gradcam_base64)
      : row.reportAssets?.ensembleGradcamDataUrl);

  const inputUrl = row.reportAssets?.inputImageDataUrl
    ?? (row.modality === "mri" && row.mriData?.preview?.cleaned_base64
      ? pngBase64ToDataUrl(row.mriData.preview.cleaned_base64)
      : null);

  const modelRows = row.modality === "xray" && row.xrayData
    ? buildXrayModelRows(row.xrayData)
    : row.modality === "mri" && row.mriData
      ? buildMriModelRows(row.mriData)
      : [];

  return (
    <div className="p-4 border-t space-y-3">
      <div className="flex items-start gap-3 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-background border font-semibold">
              {row.modality === "xray" ? "X-Ray ensemble" : "MRI pipeline"}
            </span>
            <GradeBadge grade={row.grade} />
            <ConfidenceGauge value={row.confidence} />
            <span className="text-[10px] text-muted-foreground">{row.modelUsed}</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            {row.region}{row.view ? ` · ${row.view}` : ""} · {row.inputFileName}
          </p>
          {row.mriData && (
            <p className="text-[10px] text-muted-foreground mt-1">{formatVolumeMeta(row.mriData)}</p>
          )}
          {row.mriData?.category_feedback?.length ? (
            <ul className="mt-2 space-y-0.5">
              {row.mriData.category_feedback.map((f, i) => (
                <li key={i} className="text-[11px] text-muted-foreground flex gap-1.5">
                  <FlaskConical className="w-3 h-3 text-primary flex-shrink-0 mt-0.5" />{f}
                </li>
              ))}
            </ul>
          ) : (
            <ul className="mt-2 space-y-0.5">
              {row.findings.slice(0, 4).map((f, i) => (
                <li key={i} className="text-[11px] text-muted-foreground">• {f}</li>
              ))}
            </ul>
          )}
        </div>
        <div className="flex gap-2 flex-shrink-0">
          {inputUrl && (
            <div className="w-20 h-20 rounded-lg border overflow-hidden bg-muted">
              <img src={inputUrl} alt="Input" className="w-full h-full object-cover" />
            </div>
          )}
          {previewUrl && (
            <div className="w-20 h-20 rounded-lg border overflow-hidden bg-muted">
              <img src={previewUrl} alt="Grad-CAM" className="w-full h-full object-cover" />
            </div>
          )}
        </div>
      </div>
      {modelRows.length > 0 && (
        <div className="rounded-lg border overflow-hidden">
          <div className="px-3 py-2 bg-muted/40 border-b text-[10px] uppercase font-semibold tracking-wider text-muted-foreground">
            Model outputs
          </div>
          <div className="divide-y">
            {modelRows.map(m => (
              <div key={m.id} className="px-3 py-2 flex items-center justify-between gap-2 text-xs">
                <span className="font-medium truncate">{m.name}</span>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {m.gradeDisplay ?? (m.grade != null ? <GradeBadge grade={m.grade} /> : null)}
                  <span className="text-[11px] tabular-nums text-muted-foreground">
                    {m.confidenceDisplay ?? `${m.confidence.toFixed(1)}%`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {row.modality === "xray" && row.xrayData && !row.xrayData.is_reliable && (
        <p className="text-[11px] text-warning flex items-center gap-1">
          <AlertTriangle className="w-3.5 h-3.5" /> Low ensemble agreement — review individual models.
        </p>
      )}
      {row.modality === "mri" && row.mriData && !row.mriData.is_reliable && (
        <p className="text-[11px] text-warning flex items-center gap-1">
          <AlertTriangle className="w-3.5 h-3.5" /> Low DeiT-S confidence — review multi-label categories.
        </p>
      )}
    </div>
  );
}

function ProcessingScreen({
  patients,
  cohortInputs,
  cohortFiles,
  applyAnalysisResult,
  onComplete,
  onCancel,
}: {
  patients: Patient[];
  cohortInputs: Map<string, CohortInputEntry>;
  cohortFiles: CohortFilesMap;
  applyAnalysisResult: ReturnType<typeof usePatients>["applyAnalysisResult"];
  onComplete: (results: Map<string, BatchPatientResult>) => void;
  onCancel: () => void;
}) {
  const totalEta = useMemo(() => patients.reduce((s, p) => s + estimateSecondsForPatient(p), 0), [patients]);
  const [remaining, setRemaining] = useState(totalEta);
  const [progressMap, setProgressMap] = useState<Map<string, PatientProgress>>(() => {
    const m = new Map<string, PatientProgress>();
    patients.forEach((p, i) => m.set(p.id, { patientId: p.id, progress: 0, stage: i === 0 ? "Starting…" : "Queued", status: i === 0 ? "processing" : "queued" }));
    return m;
  });
  const [fatalError, setFatalError] = useState<string | null>(null);

  useEffect(() => {
    const t = setInterval(() => setRemaining(r => Math.max(0, r - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const results = new Map<string, BatchPatientResult>();

    const run = async () => {
      for (let i = 0; i < patients.length; i++) {
        if (cancelled) return;
        const p = patients[i];
        const mods = getPatientModalities(p);
        const files = cohortFiles.get(p.id);
        const region = p.scans[0]?.region ?? "Knee";
        const views: Partial<Record<Modality, string>> = {
          xray: p.scans.find(s => s.modality === "xray")?.view ?? "AP",
          mri: p.scans.find(s => s.modality === "mri")?.view ?? "Axial",
        };

        setProgressMap(prev => {
          const next = new Map(prev);
          next.set(p.id, { patientId: p.id, progress: 5, stage: "Starting analysis…", status: "processing" });
          return next;
        });

        try {
          const result = await runPatientBatchAnalysis(
            p.id,
            mods,
            cohortInputs.get(p.id),
            files,
            region,
            views,
            {},
            stage => {
              if (cancelled) return;
              setProgressMap(prev => {
                const next = new Map(prev);
                const cur = next.get(p.id);
                next.set(p.id, {
                  patientId: p.id,
                  progress: Math.min(95, (cur?.progress ?? 5) + 8),
                  stage,
                  status: "processing",
                });
                return next;
              });
            },
          );

          results.set(p.id, result);
          const payload = batchResultToApplyPayload(result);
          if (payload) applyAnalysisResult(p.id, payload);

          setProgressMap(prev => {
            const next = new Map(prev);
            next.set(p.id, {
              patientId: p.id,
              progress: 100,
              stage: result.perModality.some(r => r.error) ? "Completed with warnings" : "Complete",
              status: result.perModality.every(r => r.error) ? "error" : "completed",
            });
            if (i + 1 < patients.length) {
              const np = patients[i + 1];
              next.set(np.id, { patientId: np.id, progress: 0, stage: "Starting…", status: "processing" });
            }
            return next;
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Analysis failed";
          setProgressMap(prev => {
            const next = new Map(prev);
            next.set(p.id, { patientId: p.id, progress: 100, stage: msg, status: "error" });
            return next;
          });
          if (i === 0 && patients.length === 1) setFatalError(msg);
        }
      }

      if (!cancelled) {
        setTimeout(() => onComplete(results), 400);
      }
    };

    run();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const completedCount = Array.from(progressMap.values()).filter(p => p.status === "completed").length;
  const overallProgress = patients.length
    ? Array.from(progressMap.values()).reduce((s, p) => s + p.progress, 0) / patients.length
    : 0;

  return (
    <motion.div initial={false} animate={{ opacity: 1 }} className="flex-1 min-h-0 overflow-auto">
      <div className="max-w-lg mx-auto px-4 sm:px-6 py-10 sm:py-12 flex flex-col items-center">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-2">Estimated time remaining</p>
        <p className="text-5xl font-light tabular-nums tracking-tight mb-1">
          {Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, "0")}
        </p>
        <p className="text-xs text-muted-foreground mb-6">
          Analyzing · {completedCount} of {patients.length} complete · {Math.round(overallProgress)}%
        </p>

        <div className="w-full h-1 bg-muted rounded-full overflow-hidden mb-6">
          <motion.div className="h-full bg-primary" animate={{ width: `${overallProgress}%` }} transition={{ duration: 0.3 }} />
        </div>

        <div className="w-full space-y-2 mb-6 max-h-[min(42vh,360px)] overflow-y-auto pr-0.5">
          {patients.map(p => {
            const prog = progressMap.get(p.id);
            return (
              <div key={p.id} className="flex items-center gap-3 text-sm">
                <div className="w-4 flex-shrink-0">
                  {prog?.status === "completed"
                    ? <Check className="w-4 h-4 text-success" />
                    : prog?.status === "error"
                    ? <AlertTriangle className="w-4 h-4 text-destructive" />
                    : prog?.status === "processing"
                    ? <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />
                    : <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 mx-auto" />}
                </div>
                <span className="truncate flex-1 text-foreground/90">{p.name}</span>
                <span className="text-[11px] text-muted-foreground truncate max-w-[200px]">
                  {prog?.status === "queued" ? "Queued" : prog?.stage}
                </span>
              </div>
            );
          })}
        </div>

        {fatalError && (
          <p className="text-xs text-destructive mb-4 text-center max-w-sm">{fatalError}</p>
        )}

        <button onClick={onCancel} className="text-xs text-muted-foreground hover:text-destructive transition-colors">
          Cancel
        </button>
      </div>
    </motion.div>
  );
}

// ============================================================
// Batch review — doctor confirm or override after cohort analysis
// ============================================================
type BatchReviewStatus = "pending" | "confirmed" | "overridden";

type BatchReviewRecord = {
  status: BatchReviewStatus;
  finalGrade?: number;
};

function buildBatchConfirmPayload(
  p: Patient,
  batchResult: BatchPatientResult | undefined,
  cohortInput: CohortInputEntry | undefined,
  grade: number,
  analysis: JointAnalysis,
  doctorOverride = false,
  overrideNotes?: string,
): ConfirmDiagnosisPayload {
  const payload = batchResult ? batchResultToApplyPayload(batchResult) : null;
  const mods = getPatientModalities(p);
  const primaryMod = payload?.modality ?? mods[0] ?? p.modality;
  const inputNames = mods.map(m => cohortInput?.[m]?.fileName).filter(Boolean).join(", ");
  const scan = p.scans.find(s => s.modality === primaryMod) ?? p.scans[0];
  const modelUsed = payload?.modelUsed ?? (primaryMod === "xray" ? "Ensemble (8 models)" : "MACS-Net + DeiT-S");
  const findings = doctorOverride
    ? getFindingsForGrade(primaryMod, grade)
    : (payload?.findings ?? getFindingsForGrade(primaryMod, grade));
  const aiConfidence = analysis.finalConfidence;
  const diagnosisSummary = doctorOverride
    ? buildDiagnosisSummary(grade, aiConfidence, findings, modelUsed)
    : (payload?.diagnosisSummary ?? buildDiagnosisSummary(grade, aiConfidence, findings, modelUsed));

  return {
    grade,
    aiConfidence,
    findings,
    diagnosisSummary,
    modality: primaryMod,
    view: payload?.view ?? scan?.view ?? "",
    region: payload?.region ?? scan?.region ?? "Knee",
    inputFileName: payload?.inputFileName ?? (inputNames || `${p.name} scan`),
    modelUsed,
    inputImageDataUrl: payload?.inputImageDataUrl,
    ensembleGradcamDataUrl: payload?.ensembleGradcamDataUrl,
    modelResults: payload?.modelResults,
    modalitySnapshots: payload?.modalitySnapshots,
    doctorOverride,
    overrideNotes,
  };
}

function BatchPatientReviewPanel({
  patient,
  batchResult,
  cohortInput,
  analysis,
  review,
  onReviewComplete,
}: {
  patient: Patient;
  batchResult?: BatchPatientResult;
  cohortInput?: CohortInputEntry;
  analysis: JointAnalysis;
  review: BatchReviewRecord;
  onReviewComplete: (record: BatchReviewRecord) => void;
}) {
  const { confirmDiagnosis } = usePatients();
  const [showOverride, setShowOverride] = useState(false);
  const [overrideGrade, setOverrideGrade] = useState<number>(analysis.finalGrade);
  const [overrideNotes, setOverrideNotes] = useState("");

  const submitReview = (grade: number, doctorOverride: boolean, notes?: string) => {
    confirmDiagnosis(
      patient.id,
      buildBatchConfirmPayload(patient, batchResult, cohortInput, grade, analysis, doctorOverride, notes),
    );
    onReviewComplete({
      status: doctorOverride ? "overridden" : "confirmed",
      finalGrade: grade,
    });
    setShowOverride(false);
    toast.success(doctorOverride ? "Override saved" : "Diagnosis confirmed", {
      description: doctorOverride
        ? `${patient.name} — report updated to KL Grade ${grade} (doctor override).`
        : `${patient.name} — AI diagnosis confirmed at KL Grade ${grade}.`,
    });
  };

  if (review.status === "confirmed") {
    return (
      <div className="p-2.5 rounded-lg border border-success/30 bg-success/5 flex items-start gap-2.5">
        <CheckCircle2 className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-medium text-success">Diagnosis confirmed</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            AI grade KL {review.finalGrade} accepted · report saved for {patient.name}.
          </p>
        </div>
      </div>
    );
  }

  if (review.status === "overridden") {
    return (
      <div className="p-2.5 rounded-lg border border-warning/30 bg-warning/5 flex items-start gap-2.5">
        <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-medium text-warning">Doctor override applied</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            Final grade KL {review.finalGrade} · report updated for {patient.name}.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-primary/25 bg-primary/5 overflow-hidden">
      <div className="p-3 flex flex-col sm:flex-row sm:items-center gap-3">
        <div className="flex items-start gap-2.5 flex-1 min-w-0">
          <ShieldCheck className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium">Physician review required</p>
            <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">
              AI recommends <span className="font-medium text-foreground">KL Grade {analysis.finalGrade}</span> at{" "}
              {analysis.finalConfidence}% confidence. Confirm if you agree, or override with your clinical grade.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            type="button"
            onClick={() => submitReview(analysis.finalGrade, false)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-success text-success-foreground text-xs font-medium hover:bg-success/90 transition-colors"
          >
            <Check className="w-3.5 h-3.5" /> Confirm AI diagnosis
          </button>
          <button
            type="button"
            onClick={() => {
              setOverrideGrade(analysis.finalGrade);
              setShowOverride(v => !v);
            }}
            className={cn(
              "inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-xs font-medium transition-colors",
              showOverride
                ? "bg-warning/10 border-warning text-warning"
                : "text-warning hover:bg-warning hover:text-warning-foreground",
            )}
          >
            <X className="w-3.5 h-3.5" /> Override
          </button>
        </div>
      </div>
      <AnimatePresence>
        {showOverride && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-warning/20 bg-warning/5 overflow-hidden"
          >
            <div className="p-4">
              <p className="text-xs font-medium mb-3">Manual grade override</p>
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className="text-xs text-muted-foreground">Your grade:</span>
                {[0, 1, 2, 3, 4].map(g => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => setOverrideGrade(g)}
                    className={cn(
                      "w-9 h-9 rounded-lg text-xs font-medium transition-all",
                      overrideGrade === g ? "bg-primary text-primary-foreground" : "bg-background border text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {g}
                  </button>
                ))}
              </div>
              <textarea
                value={overrideNotes}
                onChange={e => setOverrideNotes(e.target.value)}
                placeholder="Clinical reasoning for override (optional but recommended)…"
                className="w-full px-3 py-2 rounded-lg border bg-background text-sm resize-none h-16 focus:outline-none focus:ring-2 focus:ring-ring/20"
              />
              <div className="flex justify-end gap-2 mt-3">
                <button
                  type="button"
                  onClick={() => setShowOverride(false)}
                  className="px-3 py-1.5 text-xs rounded-lg border hover:bg-muted transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => submitReview(overrideGrade, true, overrideNotes.trim() || undefined)}
                  className="px-3 py-1.5 text-xs rounded-lg bg-warning text-warning-foreground hover:bg-warning/90 transition-colors"
                >
                  Submit override
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================
// Results overview — combined findings + references for cohort
// ============================================================
function ResultsOverview({
  patients,
  cohortInputs,
  cohortResults,
  onOpenWorkspace,
  onBackToSelect,
}: {
  patients: Patient[];
  cohortInputs: Map<string, CohortInputEntry>;
  cohortResults: Map<string, BatchPatientResult>;
  onOpenWorkspace: (p: Patient) => void;
  onBackToSelect: () => void;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [previewPatient, setPreviewPatient] = useState<Patient | null>(null);
  const [reviewByPatient, setReviewByPatient] = useState<Map<string, BatchReviewRecord>>(() => {
    const m = new Map<string, BatchReviewRecord>();
    patients.forEach(p => m.set(p.id, { status: "pending" }));
    return m;
  });

  const reviewedCount = patients.filter(p => reviewByPatient.get(p.id)?.status !== "pending").length;
  const pendingCount = patients.length - reviewedCount;

  const toggle = (id: string) => setExpanded(prev => {
    const next = new Set(prev); if (next.has(id)) next.delete(id); else next.add(id); return next;
  });

  const markReviewed = (patientId: string, record: BatchReviewRecord) => {
    setReviewByPatient(prev => new Map(prev).set(patientId, record));
  };

  // Sync if doctor confirmed/overrode from workspace and returned to batch results
  useEffect(() => {
    setReviewByPatient(prev => {
      const next = new Map(prev);
      let changed = false;
      for (const p of patients) {
        if (p.report?.doctorConfirmed && next.get(p.id)?.status === "pending") {
          next.set(p.id, {
            status: p.report.doctorOverride ? "overridden" : "confirmed",
            finalGrade: p.report.finalGrade,
          });
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [patients]);
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 overflow-auto">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-full bg-success/10 flex items-center justify-center flex-shrink-0">
              <CheckCircle2 className="w-4 h-4 text-success" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-semibold">Review AI Diagnoses</h1>
              <p className="text-xs text-muted-foreground mt-0.5">
                {patients.length} patient{patients.length !== 1 ? "s" : ""} — confirm or override each diagnosis.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className={cn(
              "text-[11px] px-2 py-1 rounded-full font-medium",
              pendingCount > 0 ? "bg-warning/10 text-warning" : "bg-success/10 text-success",
            )}>
              {reviewedCount}/{patients.length} reviewed
            </span>
            <button onClick={onBackToSelect} className="px-2.5 py-1.5 rounded-lg border text-xs font-medium hover:bg-muted transition-colors">
              New cohort
            </button>
          </div>
        </div>

        {pendingCount > 0 && (
          <div className="mb-3 p-2.5 rounded-lg border border-primary/20 bg-primary/5 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              <span className="font-medium text-foreground">Action required —</span> confirm or override each patient below. Reports finalize after your decision.
            </p>
          </div>
        )}

        {pendingCount === 0 && patients.length > 0 && (
          <div className="mb-3 p-2.5 rounded-lg border border-success/30 bg-success/5 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-success flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-success font-medium">All patients reviewed — batch complete.</p>
          </div>
        )}

        <div className="space-y-2.5">
          {patients.map(p => {
            const batchResult = cohortResults.get(p.id);
            const analysis = computeJointAnalysis(p, batchResult);
            const mods = getPatientModalities(p);
            const review = reviewByPatient.get(p.id) ?? { status: "pending" as const };
            const isExpanded = expanded.has(p.id);
            return (
              <div key={p.id} className={cn(
                "border rounded-lg overflow-hidden",
                review.status === "pending" && "border-primary/20",
                review.status === "confirmed" && "border-success/30",
                review.status === "overridden" && "border-warning/30",
              )}>
                <div className="px-3 py-2.5 bg-muted/30 border-b grid grid-cols-1 md:grid-cols-[1fr_auto] gap-2.5 items-center">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-primary" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <p className="text-sm font-medium">{p.name}</p>
                        <span className="text-[10px] text-muted-foreground font-mono">{p.id}</span>
                        {review.status === "pending" && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-warning/10 text-warning font-medium">Pending</span>
                        )}
                        {review.status === "confirmed" && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">Confirmed</span>
                        )}
                        {review.status === "overridden" && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-warning/10 text-warning font-medium">Overridden</span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5 flex-wrap text-[10px] text-muted-foreground">
                        <span>{p.age}yo · BMI {p.bmi}</span>
                        {mods.includes("xray") && <span className="uppercase px-1 py-0.5 rounded bg-background border font-medium">X-Ray</span>}
                        {mods.includes("mri") && <span className="uppercase px-1 py-0.5 rounded bg-background border font-medium">MRI</span>}
                        {mods.length > 1 && <span className="uppercase px-1 py-0.5 rounded bg-primary/10 text-primary font-medium">Joint</span>}
                        {!isExpanded && (
                          <span className="text-foreground/80">
                            · KL {analysis.finalGrade} · {analysis.finalConfidence}%
                            {analysis.perModality.length > 1 && (
                              <span className={analysis.agreement === "concordant" ? " text-success" : " text-warning"}>
                                {" "}· {analysis.agreement === "concordant" ? "concordant" : "discordant"}
                              </span>
                            )}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0 flex-wrap md:justify-end">
                    <div className="flex items-center gap-1.5">
                      <GradeBadge grade={analysis.finalGrade} />
                      <ConfidenceGauge value={analysis.finalConfidence} />
                    </div>
                    <button
                      onClick={() => setPreviewPatient(p)}
                      className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md border text-[11px] font-medium hover:bg-muted transition-colors"
                    >
                      <Scan className="w-3 h-3" /> Inputs
                    </button>
                    <button
                      onClick={() => onOpenWorkspace(p)}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-primary text-primary-foreground text-[11px] font-medium hover:bg-primary/90 transition-colors"
                    >
                      Workspace<ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                <div className="px-3 py-2.5 space-y-2">
                  <BatchPatientReviewPanel
                    patient={p}
                    batchResult={batchResult}
                    cohortInput={cohortInputs.get(p.id)}
                    analysis={analysis}
                    review={review}
                    onReviewComplete={record => markReviewed(p.id, record)}
                  />
                  <button
                    type="button"
                    onClick={() => toggle(p.id)}
                    className="inline-flex items-center gap-1 text-[11px] font-medium text-primary hover:underline"
                  >
                    <ChevronRight className={cn("w-3 h-3 transition-transform", isExpanded && "rotate-90")} />
                    {isExpanded ? "Hide clinical details & model outputs" : "Show clinical details & model outputs"}
                  </button>
                  {isExpanded && (
                    <div className="space-y-2 pt-0.5">
                      <ClinicalInterpretation patient={p} analysis={analysis} batchResult={batchResult} compact />
                    <div className="rounded-lg border bg-muted/20 overflow-hidden">
                      {batchResult?.perModality.map(row => (
                        <BatchModalityDetail key={row.modality} row={row} />
                      ))}
                      {!batchResult && (
                        <div className="p-3 text-xs text-muted-foreground">No batch API results in session — re-run analysis.</div>
                      )}
                      {cohortInputs.get(p.id) && getPatientModalities(p).map(mod => {
                        const upload = cohortInputs.get(p.id)?.[mod];
                        if (!upload) return null;
                        return (
                          <div key={mod} className="p-3 text-xs flex items-center gap-3 border-t">
                            <Upload className="w-4 h-4 text-primary flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <p className="font-medium">{mod === "xray" ? "X-Ray" : "MRI"} uploaded input</p>
                              <p className="text-[11px] text-muted-foreground truncate">{upload.fileName}</p>
                            </div>
                            {upload.previewUrl && (
                              <div className="w-12 h-12 rounded border overflow-hidden flex-shrink-0">
                                <img src={upload.previewUrl} alt="" className="w-full h-full object-cover" />
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <PatientPreviewDialog patient={previewPatient} onClose={() => setPreviewPatient(null)} />
    </motion.div>
  );
}

// ============================================================
// Diagnostic Workspace
// ============================================================
function DiagnosticWorkspace({
  patientId,
  onBack,
  initialBatchResult,
  cohortInputEntry,
  cohortFiles,
}: {
  patientId: string;
  onBack: () => void;
  initialBatchResult?: BatchPatientResult;
  cohortInputEntry?: CohortInputEntry;
  cohortFiles?: Partial<Record<Modality, File>>;
}) {
  const { getPatient, confirmDiagnosis, applyAnalysisResult } = usePatients();
  const navigate = useNavigate();
  const patient = getPatient(patientId);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const [activeModality, setActiveModality] = useState<Modality>(() => getPatient(patientId)?.modality ?? "xray");
  const views = activeModality === "xray" ? xrayViews : mriViews;

  const [showGradCAM, setShowGradCAM] = useState(true);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [zoom, setZoom] = useState(100);
  const [activeTool, setActiveTool] = useState("select");
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [measurements, setMeasurements] = useState<{ id: string; x1: number; y1: number; x2: number; y2: number }[]>([]);
  const [measureStart, setMeasureStart] = useState<{ x: number; y: number } | null>(null);
  const [annotations, setAnnotations] = useState<{ id: string; x: number; y: number; label: string }[]>([]);
  const [drawingPaths, setDrawingPaths] = useState<{ id: string; points: { x: number; y: number }[]; color: string; size: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentDrawPath, setCurrentDrawPath] = useState<{ x: number; y: number }[]>([]);
  const [drawColor, setDrawColor] = useState("#ef4444");
  const [drawSize, setDrawSize] = useState(2);
  const [textBoxes, setTextBoxes] = useState<{ id: string; x: number; y: number; text: string; color: string; fontSize: number; rotation: number; width: number }[]>([]);
  const [editingTextId, setEditingTextId] = useState<string | null>(null);
  const [selectedTextId, setSelectedTextId] = useState<string | null>(null);
  const [textColor, setTextColor] = useState("#ffffff");
  const [textFontSize, setTextFontSize] = useState(14);
  const [measureColor, setMeasureColor] = useState("#6366f1");
  const [annotateColor, setAnnotateColor] = useState("#eab308");
  const [draggingTextId, setDraggingTextId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [rotatingTextId, setRotatingTextId] = useState<string | null>(null);
  const [rotateCenter, setRotateCenter] = useState({ x: 0, y: 0 });
  const [resizingTextId, setResizingTextId] = useState<string | null>(null);
  const [resizeStartX, setResizeStartX] = useState(0);
  const [resizeStartWidth, setResizeStartWidth] = useState(0);
  const [textPlaced, setTextPlaced] = useState(false);
  const [draggingMeasurePoint, setDraggingMeasurePoint] = useState<{ measureId: string; point: "start" | "end" } | null>(null);
  const [draggingAnnotation, setDraggingAnnotation] = useState<string | null>(null);
  const [dragElementOffset, setDragElementOffset] = useState({ x: 0, y: 0 });
  const textOptionsRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<KonvaImageEditorHandle>(null);
  const sessionsRef = useRef<Partial<Record<Modality, ModalitySession>>>({});
  const uploadedFilesRef = useRef<Partial<Record<Modality, File>>>({});
  const xrayApiRunningRef = useRef(false);
  const mriApiRunningRef = useRef(false);
  const useSampleMriRef = useRef(false);
  const reportAssetsRef = useRef<Awaited<ReturnType<typeof buildReportDiagnosisAssets>> | null>(null);
  const [xrayApiData, setXrayApiData] = useState<XrayPredictResponse | null>(null);
  const [mriApiData, setMriApiData] = useState<MriPredictResponse | null>(null);
  const [mriUsesServerSample, setMriUsesServerSample] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<{ grade: number; confidence: number; findings: string[] } | null>(null);
  const [selectedModelIds, setSelectedModelIds] = useState<Set<string>>(new Set(["ensemble"]));
  const [selectedMriStageIds, setSelectedMriStageIds] = useState<Set<string>>(defaultSelectedMriStageIds());
  const [mriViewMode, setMriViewMode] = useState<MriViewMode>("gradcam");
  const [mriGallerySliceIdx, setMriGallerySliceIdx] = useState<number | null>(null);

  const penColors = [
    { id: "red", value: "#ef4444", label: "Red" },
    { id: "blue", value: "#3b82f6", label: "Blue" },
    { id: "green", value: "#22c55e", label: "Green" },
    { id: "yellow", value: "#eab308", label: "Yellow" },
    { id: "orange", value: "#f97316", label: "Orange" },
    { id: "purple", value: "#a855f7", label: "Purple" },
    { id: "cyan", value: "#06b6d4", label: "Cyan" },
    { id: "white", value: "#ffffff", label: "White" },
  ];
  const [overrideGrade, setOverrideGrade] = useState<number | null>(null);
  const [showOverridePanel, setShowOverridePanel] = useState(false);
  const [overrideNotes, setOverrideNotes] = useState("");
  const [selectedView, setSelectedView] = useState(views[0]);
  const [diagnosticStage, setDiagnosticStage] = useState<DiagnosticStage>("idle");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [stagesCompleted, setStagesCompleted] = useState<string[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [showInputConfirm, setShowInputConfirm] = useState(false);
  const [feedbackConfirmed, setFeedbackConfirmed] = useState(false);
  const batchHydratedRef = useRef(false);

  useEffect(() => {
    if (batchHydratedRef.current || !initialBatchResult) return;
    batchHydratedRef.current = true;

    const okRows = initialBatchResult.perModality.filter(r => !r.error);
    if (!okRows.length) return;

    for (const row of okRows) {
      const previewUrl = cohortInputEntry?.[row.modality]?.previewUrl ?? null;
      const stageIds = (row.modality === "xray" ? xrayStages : mriStages)
        .map(s => s.id)
        .filter(id => id !== "complete");
      const session: ModalitySession = {
        fileName: row.inputFileName,
        imageUrl: previewUrl,
        stage: "complete",
        stagesCompleted: stageIds,
        currentStageIndex: stageIds.length,
        uploadProgress: 100,
        feedbackConfirmed: false,
        selectedView: row.view,
      };
      sessionsRef.current[row.modality] = session;
      if (cohortFiles?.[row.modality]) {
        uploadedFilesRef.current[row.modality] = cohortFiles[row.modality];
      }
      if (row.modality === "xray" && row.xrayData) {
        setXrayApiData(row.xrayData);
        setSelectedModelIds(defaultSelectedModelIds(row.xrayData));
      }
      if (row.modality === "mri" && row.mriData) {
        setMriApiData(row.mriData);
        setSelectedMriStageIds(defaultSelectedMriStageIds());
        setMriGallerySliceIdx(row.mriData.primary_slice_idx ?? row.mriData.preview?.center_slice_idx ?? null);
        setMriViewMode("gradcam");
      }
      if (row.reportAssets) reportAssetsRef.current = row.reportAssets;
    }

    const primary = okRows[0];
    setActiveModality(primary.modality);
    setSelectedView(primary.view);
    setUploadedFileName(primary.inputFileName);
    setUploadedImageUrl(cohortInputEntry?.[primary.modality]?.previewUrl ?? null);
    setAnalysisResult({ grade: primary.grade, confidence: primary.confidence, findings: primary.findings });
    setDiagnosticStage("complete");
    setStagesCompleted(
      (primary.modality === "xray" ? xrayStages : mriStages).map(s => s.id).filter(id => id !== "complete"),
    );
    setCurrentStageIndex((primary.modality === "xray" ? xrayStages : mriStages).length - 1);
    setUploadProgress(100);
    setFeedbackConfirmed(false);
  }, [initialBatchResult, cohortInputEntry, cohortFiles]);

  const toolCursor = activeTool === "pan" ? (isPanning ? "grabbing" : "grab")
    : activeTool === "measure" ? "crosshair" 
    : activeTool === "annotate" ? "crosshair" 
    : activeTool === "draw" ? "crosshair" 
    : activeTool === "text" ? (textPlaced ? "default" : "text")
    : "default";

  const currentScan = patient?.scans.find(s => s.modality === activeModality && s.view === selectedView) || patient?.scans[0];
  const stages = activeModality === "xray" ? xrayStages : mriStages;
  const result =
    analysisResult && (activeModality === "xray" || activeModality === "mri")
      ? analysisResult
      : mockResults[activeModality];
  const modelRows: ModelPerformanceRow[] = useMemo(() => {
    if (activeModality === "xray" && xrayApiData) return buildXrayModelRows(xrayApiData);
    if (activeModality === "mri" && mriApiData) return buildMriModelRows(mriApiData);
    return modelPerformance[activeModality].map(row => ({
      id: row.id,
      name: row.name,
      grade: row.grade,
      confidence: row.confidence,
      gradcamUrl: row.gradcamUrl,
    }));
  }, [activeModality, xrayApiData, mriApiData]);

  const mriPreviewDisplayUrl = mriApiData
    ? galleryImageForMode(getActiveGallerySlice(mriApiData, mriGallerySliceIdx), mriApiData.preview, mriViewMode)
    : null;
  const mriInputDisplayUrl = mriApiData
    ? galleryImageForMode(getActiveGallerySlice(mriApiData, mriGallerySliceIdx), mriApiData.preview, "raw")
    : null;
  const mriActiveSlice = mriApiData ? getActiveGallerySlice(mriApiData, mriGallerySliceIdx) : null;

  const editorDisplayUrl =
    uploadedImageUrl
    ?? (activeModality === "mri" ? mriInputDisplayUrl : null)
    ?? (activeModality === "xray" ? reportAssetsRef.current?.inputImageDataUrl : null)
    ?? null;

  const previewCamViews = useMemo(() => {
    if (!xrayApiData) return [];
    return buildGradcamViewItems(xrayApiData, selectedModelIds);
  }, [xrayApiData, selectedModelIds]);

  const primaryPreviewView = previewCamViews[0] ?? null;

  const toggleModelSelection = (id: string) => {
    setSelectedModelIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const selectOnlyModel = (id: string) => setSelectedModelIds(new Set([id]));
  const toggleMriStageSelection = (id: string) => {
    setSelectedMriStageIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };
  const selectOnlyMriStage = (id: string) => setSelectedMriStageIds(new Set([id]));
  const modelUsed = activeModality === "xray"
    ? `Ensemble · ${xrayApiData?.model_count ?? 8} models`
    : mriApiData
      ? "MACS-Net + DeiT-S"
      : "MACS-Net + DeiT-S";
  const scanRegion = currentScan?.region ?? "Knee";

  const handleModalitySwitch = (mod: Modality) => {
    sessionsRef.current[activeModality] = {
      fileName: uploadedFileName,
      imageUrl: uploadedImageUrl,
      stage: diagnosticStage,
      stagesCompleted,
      currentStageIndex,
      uploadProgress,
      feedbackConfirmed,
      selectedView,
    };
    const next = sessionsRef.current[mod] ?? emptyModalitySession(mod);
    setActiveModality(mod);
    setSelectedView(next.selectedView);
    setUploadedFileName(next.fileName);
    setUploadedImageUrl(next.imageUrl);
    setDiagnosticStage(next.stage);
    setStagesCompleted(next.stagesCompleted);
    setCurrentStageIndex(next.currentStageIndex);
    setUploadProgress(next.uploadProgress);
    setFeedbackConfirmed(next.feedbackConfirmed);
    setActiveTool("select");
  };

  const snapshotActiveSession = (): ModalitySession => ({
    fileName: uploadedFileName,
    imageUrl: uploadedImageUrl,
    stage: diagnosticStage,
    stagesCompleted,
    currentStageIndex,
    uploadProgress,
    feedbackConfirmed,
    selectedView,
  });

  const getSessionForMod = (mod: Modality): ModalitySession => {
    if (mod === activeModality) return snapshotActiveSession();
    return sessionsRef.current[mod] ?? emptyModalitySession(mod);
  };

  const startDiagnosticFlow = useCallback((fileName: string) => {
    setUploadedFileName(fileName);
    setDiagnosticStage("uploading");
    setCurrentStageIndex(0);
    setStagesCompleted([]);
    setUploadProgress(0);
  }, []);

  useEffect(() => {
    if (xrayApiRunningRef.current || mriApiRunningRef.current) return;
    // Only advance the mock pipeline after explicit confirm (runAnalysis), not while input is "ready"
    if (diagnosticStage === "idle" || diagnosticStage === "ready" || diagnosticStage === "complete") return;
    const currentStage = stages[currentStageIndex];
    if (!currentStage) return;
    if (currentStage.id === "uploading") {
      const interval = setInterval(() => {
        setUploadProgress(prev => prev >= 100 ? (clearInterval(interval), 100) : prev + Math.random() * 15 + 5);
      }, 200);
      const timer = setTimeout(() => {
        clearInterval(interval);
        setUploadProgress(100);
        setStagesCompleted(prev => [...prev, currentStage.id]);
        setCurrentStageIndex(prev => prev + 1);
        if (currentStageIndex + 1 < stages.length) setDiagnosticStage(stages[currentStageIndex + 1].id);
      }, currentStage.duration);
      return () => { clearInterval(interval); clearTimeout(timer); };
    }
    if (currentStage.id === "complete") { setDiagnosticStage("complete"); return; }
    const timer = setTimeout(() => {
      setStagesCompleted(prev => [...prev, currentStage.id]);
      setCurrentStageIndex(prev => prev + 1);
      if (currentStageIndex + 1 < stages.length) setDiagnosticStage(stages[currentStageIndex + 1].id);
    }, currentStage.duration);
    return () => clearTimeout(timer);
  }, [diagnosticStage, currentStageIndex, stages]);

  const handleFileSelect = () => fileInputRef.current?.click();
  const processFile = (file: File) => {
    if (!isValidModalityFile(file, activeModality)) {
      toast.error(`Invalid file type for ${activeModality === "xray" ? "X-Ray" : "MRI"}. Check supported formats.`);
      return;
    }
    if (activeModality === "mri") {
      useSampleMriRef.current = false;
      setMriUsesServerSample(false);
    }
    const prev = getSessionForMod(activeModality);
    if (prev.imageUrl) URL.revokeObjectURL(prev.imageUrl);
    uploadedFilesRef.current[activeModality] = file;
    setUploadedFileName(file.name);
    setDiagnosticStage("ready");
    setStagesCompleted([]);
    setCurrentStageIndex(0);
    setUploadProgress(0);
    setFeedbackConfirmed(false);
    const previewUrl = file.type.startsWith("image/") ? URL.createObjectURL(file) : null;
    setUploadedImageUrl(previewUrl);
    sessionsRef.current[activeModality] = {
      fileName: file.name,
      imageUrl: previewUrl,
      stage: "ready",
      stagesCompleted: [],
      currentStageIndex: 0,
      uploadProgress: 0,
      feedbackConfirmed: false,
      selectedView,
    };
  };
  const requiredMods = patient ? getPatientModalities(patient) : [];
  const activeModReady = useMemo(() => {
    const s = getSessionForMod(activeModality);
    if (!s.fileName || s.stage !== "ready") return false;
    if (activeModality === "mri") {
      return mriUsesServerSample || !!uploadedFilesRef.current.mri;
    }
    return !!uploadedFilesRef.current.xray;
  }, [activeModality, uploadedFileName, diagnosticStage, mriUsesServerSample]);
  const allRequiredReady = requiredMods.every(mod => {
    const s = getSessionForMod(mod);
    const ready = !!s.fileName && s.stage === "ready";
    if (mod === "mri") return ready && (mriUsesServerSample || !!uploadedFilesRef.current.mri);
    return ready && !!uploadedFilesRef.current[mod];
  });
  const confirmUploads = requiredMods
    .map(mod => {
      const s = getSessionForMod(mod);
      if (!s.fileName) return null;
      return { modality: mod, view: s.selectedView, fileName: s.fileName, previewUrl: s.imageUrl };
    })
    .filter((u): u is NonNullable<typeof u> => u != null);
  const handleLoadSampleMri = async () => {
    const health = await fetchBackboneHealth();
    if (!health?.mri_sample_available) {
      toast.error("Sample MRI not found on backbone", {
        description: "Place Effusion.nii.gz in the backbone/ folder.",
      });
      return;
    }
    const filename = health.mri_sample_filename ?? "Effusion.nii.gz";
    useSampleMriRef.current = true;
    setMriUsesServerSample(true);
    setActiveModality("mri");
    const mriView = sessionsRef.current.mri?.selectedView ?? mriViews[0];
    setSelectedView(mriView);
    delete uploadedFilesRef.current.mri;
    setUploadedFileName(filename);
    setUploadedImageUrl(null);
    setDiagnosticStage("ready");
    setStagesCompleted([]);
    setCurrentStageIndex(0);
    setUploadProgress(0);
    setFeedbackConfirmed(false);
    sessionsRef.current.mri = {
      fileName: filename,
      imageUrl: null,
      stage: "ready",
      stagesCompleted: [],
      currentStageIndex: 0,
      uploadProgress: 0,
      feedbackConfirmed: false,
      selectedView: mriView,
    };
    toast.success("Sample MRI loaded", {
      description: `${filename} on server — analysis will skip upload.`,
    });
  };

  const handleStartAnalysis = () => {
    if (!activeModReady) {
      toast.error(`Upload or load a ${activeModality === "xray" ? "X-Ray" : "MRI"} scan before analysis.`);
      return;
    }
    setShowInputConfirm(true);
  };
  const runXrayPrediction = useCallback(async (file: File) => {
    xrayApiRunningRef.current = true;
    setXrayApiData(null);
    setAnalysisResult(null);
    setSelectedModelIds(new Set(["ensemble"]));
    setSelectedMriStageIds(defaultSelectedMriStageIds());
    setUploadedFileName(file.name);
    setUploadProgress(0);
    setStagesCompleted([]);
    setCurrentStageIndex(0);
    setDiagnosticStage("uploading");
    try {
      const data = await predictXray(file, "all", (pct) => {
        setUploadProgress(pct);
        if (pct >= 100) {
          setStagesCompleted(["uploading", "preprocessing"]);
          setCurrentStageIndex(2);
          setDiagnosticStage("inference");
        }
      });
      setXrayApiData(data);
      const parsed = xrayResponseToResult(data);
      setAnalysisResult(parsed);
      setSelectedModelIds(defaultSelectedModelIds(data));
      setStagesCompleted(stages.filter(s => s.id !== "complete").map(s => s.id));
      setCurrentStageIndex(stages.length - 1);
      setDiagnosticStage("complete");
      const assets = await buildReportDiagnosisAssets(
        uploadedImageUrl,
        data,
        parsed.grade,
        parsed.confidence,
        parsed.findings,
        `Ensemble · ${data.model_count ?? Object.keys(data.individual_results).length} models`,
      );
      reportAssetsRef.current = assets;
      applyAnalysisResult(patientId, {
        grade: parsed.grade,
        aiConfidence: parsed.confidence,
        findings: parsed.findings,
        diagnosisSummary: assets.diagnosisSummary,
        modality: "xray",
        view: sessionsRef.current.xray?.selectedView ?? xrayViews[0],
        region: scanRegion,
        inputFileName: file.name,
        modelUsed: `Ensemble · ${data.model_count ?? Object.keys(data.individual_results).length} models`,
        inputImageDataUrl: assets.inputImageDataUrl,
        ensembleGradcamDataUrl: assets.ensembleGradcamDataUrl,
        modelResults: assets.modelResults,
      });
      if (!data.is_reliable) {
        toast.warning("Low ensemble confidence", { description: "Review findings before confirming." });
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "X-ray analysis failed", {
        description: "Could not reach the analysis server. Please try again.",
      });
      setDiagnosticStage("ready");
      setStagesCompleted([]);
      setCurrentStageIndex(0);
    } finally {
      xrayApiRunningRef.current = false;
    }
  }, [patientId, applyAnalysisResult, scanRegion, stages, uploadedImageUrl]);

  const runMriPrediction = useCallback(async (file: File | null, fromSample = false) => {
    mriApiRunningRef.current = true;
    setMriApiData(null);
    setAnalysisResult(null);
    setUploadProgress(fromSample ? 100 : 0);
    setStagesCompleted(fromSample ? ["uploading", "preprocessing"] : []);
    setCurrentStageIndex(fromSample ? 2 : 0);
    setDiagnosticStage(fromSample ? "artifact-removal" : "uploading");
    const displayName = fromSample ? (uploadedFileName || "Effusion.nii.gz") : file!.name;
    if (!fromSample) setUploadedFileName(displayName);
    try {
      const data = fromSample
        ? await predictMriSample()
        : await predictMri(file!, undefined, (pct) => {
            setUploadProgress(pct);
            if (pct >= 100) {
              setStagesCompleted(["uploading", "preprocessing"]);
              setCurrentStageIndex(2);
              setDiagnosticStage("artifact-removal");
            }
          });
      setMriApiData(data);
      const parsed = mriResponseToResult(data);
      setAnalysisResult(parsed);
      setSelectedMriStageIds(defaultSelectedMriStageIds());
      setMriGallerySliceIdx(data.primary_slice_idx ?? data.preview?.center_slice_idx ?? null);
      setMriViewMode("gradcam");
      setStagesCompleted(stages.filter(s => s.id !== "complete").map(s => s.id));
      setCurrentStageIndex(stages.length - 1);
      setDiagnosticStage("complete");
      const assets = await buildReportDiagnosisAssets(
        uploadedImageUrl,
        null,
        parsed.grade,
        parsed.confidence,
        parsed.findings,
        "MACS-Net + DeiT-S",
        data.preview,
      );
      reportAssetsRef.current = assets;
      applyAnalysisResult(patientId, {
        grade: parsed.grade,
        aiConfidence: parsed.confidence,
        findings: parsed.findings,
        diagnosisSummary: assets.diagnosisSummary,
        modality: "mri",
        view: sessionsRef.current.mri?.selectedView ?? mriViews[0],
        region: scanRegion,
        inputFileName: displayName,
        modelUsed: "MACS-Net + DeiT-S",
        inputImageDataUrl: assets.inputImageDataUrl,
        ensembleGradcamDataUrl: assets.ensembleGradcamDataUrl,
      });
      if (!data.is_reliable) {
        toast.warning("Low MRI classification confidence", { description: "Review multi-label findings before confirming." });
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "MRI analysis failed", {
        description: "Could not reach the analysis server. Please try again.",
      });
      setDiagnosticStage("ready");
      setStagesCompleted([]);
      setCurrentStageIndex(0);
    } finally {
      mriApiRunningRef.current = false;
    }
  }, [patientId, applyAnalysisResult, scanRegion, stages, uploadedImageUrl]);

  const runAnalysis = () => {
    setShowInputConfirm(false);
    if (activeModality === "xray") {
      const file = uploadedFilesRef.current.xray;
      if (!file) {
        toast.error("No X-ray file to analyze.");
        return;
      }
      void runXrayPrediction(file);
      return;
    }
    const mriFile = uploadedFilesRef.current.mri;
    if (useSampleMriRef.current) {
      void runMriPrediction(null, true);
      return;
    }
    if (!mriFile) {
      toast.error("No MRI file to analyze.");
      return;
    }
    void runMriPrediction(mriFile);
  };
  const buildConfirmPayload = (grade: number, doctorOverride = false, notes?: string) => {
    const assets = reportAssetsRef.current;
    return {
      grade,
      aiConfidence: result.confidence,
      findings: result.findings,
      diagnosisSummary: assets?.diagnosisSummary ?? `KL Grade ${grade} · ${result.confidence}% confidence`,
      modality: activeModality,
      view: selectedView,
      region: scanRegion,
      inputFileName: confirmUploads.map(u => `${u.modality === "xray" ? "X-Ray" : "MRI"}: ${u.fileName}`).join(" · ") || uploadedFileName,
      modelUsed,
      inputImageDataUrl: assets?.inputImageDataUrl,
      ensembleGradcamDataUrl: assets?.ensembleGradcamDataUrl,
      modelResults: assets?.modelResults,
      doctorOverride,
      overrideNotes: notes,
    };
  };
  const handleAgree = () => {
    confirmDiagnosis(patientId, buildConfirmPayload(result.grade));
    setFeedbackConfirmed(true);
    toast.success("Report updated", { description: "Patient report saved with your confirmation." });
  };
  const handleOverrideSubmit = () => {
    if (overrideGrade == null) {
      toast.error("Select a grade for the override.");
      return;
    }
    confirmDiagnosis(patientId, buildConfirmPayload(overrideGrade, true, overrideNotes));
    setFeedbackConfirmed(true);
    setShowOverridePanel(false);
    toast.success("Report updated", { description: `Override applied — Grade ${overrideGrade} saved to report.` });
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) processFile(file); };
  const handleDrop = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); const file = e.dataTransfer.files?.[0]; if (file) processFile(file); };
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const persistedAnalysisRef = useRef<Partial<Record<Modality, boolean>>>({});

  useEffect(() => {
    if (diagnosticStage !== "complete" || !patient) return;
    if (persistedAnalysisRef.current[activeModality]) return;
    if (activeModality === "xray") {
      if (xrayApiData) persistedAnalysisRef.current.xray = true;
      return;
    }
    if (activeModality === "mri") {
      if (mriApiData) persistedAnalysisRef.current.mri = true;
    }
  }, [diagnosticStage, activeModality, patient, xrayApiData, mriApiData]);

  const resetDiagnostic = () => {
    if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    Object.values(sessionsRef.current).forEach(s => { if (s?.imageUrl) URL.revokeObjectURL(s.imageUrl); });
    sessionsRef.current = {};
    uploadedFilesRef.current = {};
    useSampleMriRef.current = false;
    setMriUsesServerSample(false);
    persistedAnalysisRef.current = {};
    reportAssetsRef.current = null;
    setXrayApiData(null);
    setMriApiData(null);
    setAnalysisResult(null);
    setSelectedModelIds(new Set(["ensemble"]));
    setSelectedMriStageIds(defaultSelectedMriStageIds());
    setMriGallerySliceIdx(null);
    setMriViewMode("gradcam");
    setDiagnosticStage("idle");
    setStagesCompleted([]);
    setCurrentStageIndex(0);
    setUploadProgress(0);
    setUploadedFileName("");
    setFeedbackConfirmed(false);
    setShowInputConfirm(false);
    setMeasurements([]);
    setAnnotations([]);
    setDrawingPaths([]);
    setCurrentDrawPath([]);
    setTextBoxes([]);
    setEditingTextId(null);
    setSelectedTextId(null);
    setTextPlaced(false);
    setPanOffset({ x: 0, y: 0 });
    setZoom(100);
    setActiveTool("select");
    setUploadedImageUrl(null);
  };

  useEffect(() => { if (activeTool !== "text") setTextPlaced(false); }, [activeTool]);

  const isProcessing = diagnosticStage !== "idle" && diagnosticStage !== "ready" && diagnosticStage !== "complete";

  const getRelativePos = (e: React.MouseEvent) => {
    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    const scale = zoom / 100;
    // Center of container
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    // Mouse position relative to container
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // Reverse the transform: translate then scale from center
    const imgX = ((mx - cx - panOffset.x / 4) / scale + cx) / rect.width * 100;
    const imgY = ((my - cy - panOffset.y / 4) / scale + cy) / rect.height * 100;
    return { x: imgX, y: imgY };
  };

  const handleImageMouseDown = (e: React.MouseEvent) => {
    // Konva editor handles everything when scan is loaded
    if (diagnosticStage === "complete") return;
    if (activeTool === "pan") {
      setIsPanning(true);
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    } else if (activeTool === "draw") {
      setIsDrawing(true);
      const pos = getRelativePos(e);
      setCurrentDrawPath([pos]);
    }
  };

  const handleImageMouseMove = (e: React.MouseEvent) => {
    if (diagnosticStage === "complete") return;
    if (activeTool === "pan" && isPanning) {
      setPanOffset({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    } else if (activeTool === "draw" && isDrawing) {
      const pos = getRelativePos(e);
      setCurrentDrawPath(prev => [...prev, pos]);
    } else if (draggingTextId) {
      const pos = getRelativePos(e);
      setTextBoxes(prev => prev.map(t => t.id === draggingTextId ? { ...t, x: pos.x + dragOffset.x, y: pos.y + dragOffset.y } : t));
    } else if (resizingTextId) {
      const rect = imageContainerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const dx = e.clientX - resizeStartX;
      const scale = zoom / 100;
      const newWidthPx = Math.max(40, resizeStartWidth + dx / scale);
      const newWidthPct = (newWidthPx / rect.width) * 100;
      setTextBoxes(prev => prev.map(t => t.id === resizingTextId ? { ...t, width: newWidthPct } : t));
    } else if (draggingMeasurePoint) {
      const pos = getRelativePos(e);
      setMeasurements(prev => prev.map(m => {
        if (m.id !== draggingMeasurePoint.measureId) return m;
        if (draggingMeasurePoint.point === "start") return { ...m, x1: pos.x + dragElementOffset.x, y1: pos.y + dragElementOffset.y };
        return { ...m, x2: pos.x + dragElementOffset.x, y2: pos.y + dragElementOffset.y };
      }));
    } else if (draggingAnnotation) {
      const pos = getRelativePos(e);
      setAnnotations(prev => prev.map(a => a.id === draggingAnnotation ? { ...a, x: pos.x + dragElementOffset.x, y: pos.y + dragElementOffset.y } : a));
      const rect = imageContainerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const angle = Math.atan2(my - rotateCenter.y, mx - rotateCenter.x) * (180 / Math.PI) + 90;
      setTextBoxes(prev => prev.map(t => t.id === rotatingTextId ? { ...t, rotation: Math.round(angle / 5) * 5 } : t));
    }
  };

  const handleImageMouseUp = () => {
    if (diagnosticStage === "complete") return;
    if (activeTool === "pan") setIsPanning(false);
    if (activeTool === "draw" && isDrawing) {
      setIsDrawing(false);
      if (currentDrawPath.length > 1) {
        setDrawingPaths(prev => [...prev, { id: `d${Date.now()}`, points: currentDrawPath, color: drawColor, size: drawSize }]);
      }
      setCurrentDrawPath([]);
    }
    if (draggingTextId) setDraggingTextId(null);
    if (rotatingTextId) setRotatingTextId(null);
    if (resizingTextId) setResizingTextId(null);
    if (draggingMeasurePoint) setDraggingMeasurePoint(null);
    if (draggingAnnotation) setDraggingAnnotation(null);
  };

  const handleTextDragStart = (e: React.MouseEvent, tb: typeof textBoxes[0]) => {
    e.stopPropagation();
    e.preventDefault();
    const pos = getRelativePos(e);
    setDragOffset({ x: tb.x - pos.x, y: tb.y - pos.y });
    setDraggingTextId(tb.id);
    setSelectedTextId(tb.id);
  };

  const handleRotateStart = (e: React.MouseEvent, tb: typeof textBoxes[0]) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const scale = zoom / 100;
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const tbScreenX = ((tb.x / 100 * rect.width - cx) * scale + cx + panOffset.x / 4) + rect.left;
    const tbScreenY = ((tb.y / 100 * rect.height - cy) * scale + cy + panOffset.y / 4) + rect.top;
    setRotateCenter({ x: tbScreenX - rect.left, y: tbScreenY - rect.top });
    setRotatingTextId(tb.id);
    setSelectedTextId(tb.id);
  };

  const handleResizeStart = (e: React.MouseEvent, tb: typeof textBoxes[0]) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setResizingTextId(tb.id);
    setResizeStartX(e.clientX);
    setResizeStartWidth((tb.width / 100) * rect.width);
    setSelectedTextId(tb.id);
  };

  const handleMeasurePointDrag = (e: React.MouseEvent, measureId: string, point: "start" | "end") => {
    if (activeTool !== "select") return;
    e.stopPropagation();
    e.preventDefault();
    const m = measurements.find(mm => mm.id === measureId);
    if (!m) return;
    const pos = getRelativePos(e);
    const px = point === "start" ? m.x1 : m.x2;
    const py = point === "start" ? m.y1 : m.y2;
    setDragElementOffset({ x: px - pos.x, y: py - pos.y });
    setDraggingMeasurePoint({ measureId, point });
  };

  const handleAnnotationDrag = (e: React.MouseEvent, annotationId: string) => {
    if (activeTool !== "select") return;
    e.stopPropagation();
    e.preventDefault();
    const a = annotations.find(aa => aa.id === annotationId);
    if (!a) return;
    const pos = getRelativePos(e);
    setDragElementOffset({ x: a.x - pos.x, y: a.y - pos.y });
    setDraggingAnnotation(annotationId);
  };

  const handleImageClick = (e: React.MouseEvent) => {
    if (diagnosticStage === "complete") return; // Konva editor owns interactions
    const pos = getRelativePos(e);
    if (activeTool === "zoom") {
      setZoom(prev => Math.min(200, prev + 25));
    } else if (activeTool === "measure") {
      if (!measureStart) {
        setMeasureStart(pos);
      } else {
        setMeasurements(prev => [...prev, { id: `m${Date.now()}`, x1: measureStart.x, y1: measureStart.y, x2: pos.x, y2: pos.y }]);
        setMeasureStart(null);
      }
    } else if (activeTool === "annotate") {
      const label = `A${annotations.length + 1}`;
      setAnnotations(prev => [...prev, { id: `a${Date.now()}`, x: pos.x, y: pos.y, label }]);
    } else if (activeTool === "text" && !textPlaced) {
      const newId = `t${Date.now()}`;
      setTextBoxes(prev => [...prev, { id: newId, x: pos.x, y: pos.y, text: "Text", color: textColor, fontSize: textFontSize, rotation: 0, width: 15 }]);
      setSelectedTextId(newId);
      setEditingTextId(newId);
      setTextPlaced(true);
      setTimeout(() => setActiveTool("select"), 50);
    } else if (activeTool === "select") {
      setSelectedTextId(null);
      setEditingTextId(null);
    }
  };

  // Download the annotated image
  const handleDownloadImage = useCallback(() => {
    const container = imageContainerRef.current;
    if (!container || !uploadedImageUrl) return;
    const canvas = document.createElement("canvas");
    const img = container.querySelector("img");
    if (!img) return;
    canvas.width = img.naturalWidth || 800;
    canvas.height = img.naturalHeight || 800;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    // Draw paths
    drawingPaths.forEach(dp => {
      ctx.beginPath();
      ctx.strokeStyle = dp.color;
      ctx.lineWidth = dp.size;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      dp.points.forEach((p, i) => {
        const px = (p.x / 100) * canvas.width;
        const py = (p.y / 100) * canvas.height;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });
    // Draw text boxes
    textBoxes.forEach(tb => {
      ctx.save();
      const tx = (tb.x / 100) * canvas.width;
      const ty = (tb.y / 100) * canvas.height;
      ctx.translate(tx, ty);
      ctx.rotate((tb.rotation * Math.PI) / 180);
      ctx.font = `600 ${tb.fontSize * (canvas.width / 500)}px sans-serif`;
      ctx.fillStyle = tb.color;
      ctx.textAlign = "center";
      ctx.fillText(tb.text, 0, 0);
      ctx.restore();
    });
    // Draw measurements
    measurements.forEach(m => {
      const x1 = (m.x1 / 100) * canvas.width, y1 = (m.y1 / 100) * canvas.height;
      const x2 = (m.x2 / 100) * canvas.width, y2 = (m.y2 / 100) * canvas.height;
      ctx.beginPath(); ctx.setLineDash([6, 3]); ctx.strokeStyle = "#6366f1"; ctx.lineWidth = 2;
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(x1, y1, 4, 0, Math.PI * 2); ctx.fillStyle = "#6366f1"; ctx.fill();
      ctx.beginPath(); ctx.arc(x2, y2, 4, 0, Math.PI * 2); ctx.fill();
      ctx.font = "12px sans-serif"; ctx.fillStyle = "#6366f1"; ctx.textAlign = "center";
      ctx.fillText(`${Math.round(Math.sqrt(Math.pow(m.x2 - m.x1, 2) + Math.pow(m.y2 - m.y1, 2)) * 2.5)}mm`, (x1 + x2) / 2, (y1 + y2) / 2 - 8);
    });
    // Annotations
    annotations.forEach(a => {
      const ax = (a.x / 100) * canvas.width, ay = (a.y / 100) * canvas.height;
      ctx.beginPath(); ctx.arc(ax, ay, 10, 0, Math.PI * 2); ctx.fillStyle = "#eab308"; ctx.fill();
      ctx.font = "bold 10px sans-serif"; ctx.fillStyle = "#000"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(a.label, ax, ay);
    });
    const link = document.createElement("a");
    link.download = `${patient.name.replace(/\s+/g, "_")}_annotated.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }, [uploadedImageUrl, drawingPaths, textBoxes, measurements, annotations, patient?.name]);

  if (!patient) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
        <p className="text-sm text-muted-foreground">Patient not found.</p>
        <button onClick={onBack} className="text-sm text-primary hover:underline">Back</button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Sticky top bar */}
      <div className="border-b bg-background sticky top-0 z-10 flex-shrink-0">
        <div className="h-12 flex items-center justify-between px-4 sm:px-5">
          <div className="flex items-center gap-3 min-w-0">
            <button onClick={onBack} className="p-1.5 rounded-lg hover:bg-muted transition-colors flex-shrink-0">
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium truncate">{patient.name}</span>
                <span className="text-mono text-xs text-muted-foreground hidden sm:inline">{patient.id}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Modality toggle */}
            <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
              {(["xray", "mri"] as Modality[]).map(mod => {
                const s = getSessionForMod(mod);
                const hasUpload = !!s.fileName;
                const isRequired = requiredMods.includes(mod);
                return (
                  <button key={mod} onClick={() => handleModalitySwitch(mod)} className={cn("px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1", activeModality === mod ? "bg-background text-foreground shadow-sm" : "text-muted-foreground")}>
                    {mod === "xray" ? "X-Ray" : "MRI"}
                    {isRequired && hasUpload && <CheckCircle2 className="w-3 h-3 text-success" />}
                  </button>
                );
              })}
            </div>
            <StatusBadge status={feedbackConfirmed ? "confirmed" : diagnosticStage === "complete" ? "analyzed" : patient.status} />
          </div>
        </div>
        {(requiredMods.length > 1 && (diagnosticStage === "idle" || diagnosticStage === "ready")) && (
          <div className="px-4 sm:px-5 py-2 border-t bg-primary/5 text-xs text-foreground/90">
            Joint analysis patient — upload both <span className="font-medium">X-Ray</span> and <span className="font-medium">MRI</span> scans. Switch tabs above to upload each modality.
            <span className="ml-2 text-muted-foreground">
              ({requiredMods.filter(m => { const s = getSessionForMod(m); return !!s.fileName; }).length}/{requiredMods.length} uploaded)
            </span>
          </div>
        )}
        <div className="px-4 sm:px-5 py-1.5 border-t bg-muted/30 flex items-center gap-3 text-xs text-muted-foreground overflow-x-auto">
          <span className="flex items-center gap-1 flex-shrink-0"><User className="w-3 h-3" />{patient.age}yo · {patient.gender}</span>
          <span className="flex-shrink-0">BMI {patient.bmi}</span>
          <span className="flex-shrink-0">Pain {patient.painLevel}/10</span>
          <span className="flex items-center gap-1 flex-shrink-0"><Calendar className="w-3 h-3" />{patient.lastVisit}</span>
          <span className="hidden md:inline truncate">{patient.history}</span>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-auto">
        {/* Image workspace */}
        <div className="flex flex-col lg:flex-row">
          {/* Tools — only available after analysis completes */}
          {diagnosticStage === "complete" && (
            <DiagnosticsToolbar activeTool={activeTool} setActiveTool={setActiveTool} zoom={zoom} setZoom={setZoom} setBrightness={setBrightness} setContrast={setContrast} />
          )}

          {/* Original scan panel */}
          <div className="flex-1 border-r border-b flex flex-col">
            <div className="h-10 border-b flex items-center justify-between px-4 flex-shrink-0 bg-muted/20">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-0.5 bg-background rounded-md p-0.5 border">
                  {views.map(view => (
                    <button key={view} onClick={() => setSelectedView(view)} className={cn("px-2 py-0.5 rounded text-[10px] font-medium transition-all", selectedView === view ? "bg-primary text-primary-foreground" : "text-muted-foreground")}>{view}</button>
                  ))}
                </div>
                {activeModality === "mri" && diagnosticStage === "complete" && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-medium text-muted-foreground">Raw input</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {diagnosticStage !== "idle" && diagnosticStage !== "ready" && (
                  <button onClick={resetDiagnostic} className="text-[10px] text-destructive hover:underline">Reset</button>
                )}
                <span className="text-mono text-[10px] text-muted-foreground">{zoom}%</span>
              </div>
            </div>

            {/* Image area */}
            <div
              ref={imageContainerRef}
              className={cn("aspect-square max-h-[500px] bg-foreground/[0.02] flex items-center justify-center relative overflow-hidden transition-colors", isDragging && "bg-primary/5 ring-2 ring-primary/30 ring-inset")}
              style={{ cursor: toolCursor }}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onMouseDown={handleImageMouseDown}
              onMouseMove={handleImageMouseMove}
              onMouseUp={handleImageMouseUp}
              onMouseLeave={handleImageMouseUp}
              onClick={handleImageClick}
            >
              <input ref={fileInputRef} type="file" accept={acceptStringForModality(activeModality)} className="hidden" onChange={handleFileChange} />

              <AnimatePresence mode="wait">
                {diagnosticStage === "idle" ? (
                  <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="w-72 sm:w-80 rounded-xl bg-background border shadow-sm p-5 space-y-4"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {activeModality === "mri" && (
                      <div>
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Supported formats</p>
                        <div className="flex flex-wrap gap-1">
                          {mriSupportedFormats.map(f => (
                            <span key={f} className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                              {f}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleFileSelect(); }}
                      className="w-full h-44 rounded-xl bg-foreground/5 border-2 border-dashed border-border hover:border-primary/40 hover:bg-primary/5 flex flex-col items-center justify-center gap-2 transition-all"
                    >
                      <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center">
                        <Upload className="w-6 h-6 text-muted-foreground" />
                      </div>
                      <div className="text-center px-4">
                        <p className="text-sm font-medium">Upload {activeModality === "xray" ? "X-Ray" : "MRI"} scan</p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">
                          {activeModality === "xray"
                            ? "DICOM, JPEG or PNG"
                            : "DICOM, NIfTI (.nii, .nii.gz), NRRD, .pkl, .pck, etc."}
                        </p>
                        <p className="text-[10px] text-muted-foreground/70 mt-0.5">Drag & drop or click</p>
                      </div>
                    </button>
                    {activeModality === "mri" && (
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); void handleLoadSampleMri(); }}
                        className="w-full py-2.5 rounded-lg border border-dashed border-primary/30 bg-primary/5 text-xs font-medium text-primary hover:bg-primary/10 transition-colors"
                      >
                        Use pre-loaded sample (Effusion.nii.gz)
                      </button>
                    )}
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>View: <span className="font-medium text-foreground/80">{selectedView}</span></span>
                    </div>
                  </motion.div>
                ) : diagnosticStage === "ready" ? (
                  <motion.div
                    key="ready"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex flex-col"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {uploadedImageUrl ? (
                      <img src={uploadedImageUrl} alt="Uploaded scan preview" className="flex-1 w-full object-contain" draggable={false} />
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center gap-2">
                        <FileImage className="w-12 h-12 text-muted-foreground/50" />
                        <p className="text-sm font-medium">{uploadedFileName}</p>
                        {mriUsesServerSample ? (
                          <p className="text-xs text-primary font-medium">Server sample — upload skipped</p>
                        ) : (
                          <p className="text-xs text-muted-foreground">Non-image format — preview unavailable</p>
                        )}
                      </div>
                    )}
                    <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-2 z-10">
                      <span className="text-[10px] text-white/80 bg-black/50 px-2 py-0.5 rounded truncate">{uploadedFileName}</span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => { e.stopPropagation(); handleFileSelect(); }}
                          className="text-[10px] px-2.5 py-1 rounded-lg bg-background/90 border hover:bg-muted transition-colors"
                        >
                          Change file
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleStartAnalysis(); }}
                          disabled={!activeModReady}
                          className={cn(
                            "inline-flex items-center gap-1 text-[10px] px-3 py-1 rounded-lg font-medium transition-colors",
                            activeModReady ? "bg-primary text-primary-foreground hover:bg-primary/90" : "bg-muted text-muted-foreground cursor-not-allowed"
                          )}
                        >
                          <Check className="w-3 h-3" /> Confirm input
                        </button>
                      </div>
                    </div>
                  </motion.div>
                ) : diagnosticStage === "complete" ? (
                  <motion.div key="result-image" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0">
                    {editorDisplayUrl ? (
                      <KonvaImageEditor
                        ref={editorRef}
                        imageUrl={editorDisplayUrl}
                        tool={activeTool as EditorTool}
                        brightness={brightness}
                        contrast={contrast}
                        zoom={zoom}
                        drawColor={drawColor}
                        drawSize={drawSize}
                        textColor={textColor}
                        textFontSize={textFontSize}
                        measureColor={measureColor}
                        annotateColor={annotateColor}
                        onToolChange={(t) => setActiveTool(t)}
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center p-4 bg-foreground/[0.02]">
                        <ScanImageTile
                          className="w-full max-w-md"
                          maxHeight="100%"
                          imageUrl={mriInputDisplayUrl}
                          label="Raw input"
                          sublabel={`${uploadedFileName || "MRI volume"} · ${selectedView}`}
                        />
                      </div>
                    )}
                    {/* Draw options panel */}
                    {editorDisplayUrl && activeTool === "draw" && (
                      <div className="absolute top-3 left-3 z-30 bg-background/95 backdrop-blur-sm border rounded-xl p-2.5 shadow-lg space-y-2 w-[170px]">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Pen Color</p>
                        <div className="flex flex-wrap gap-1.5">
                          {penColors.map(c => (
                            <button
                              key={c.id}
                              onClick={() => setDrawColor(c.value)}
                              className={cn("w-6 h-6 rounded-full border-2 transition-all", drawColor === c.value ? "border-foreground scale-110 shadow-sm" : "border-transparent hover:scale-105")}
                              style={{ backgroundColor: c.value }}
                              title={c.label}
                            />
                          ))}
                        </div>
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider pt-1">Size · {drawSize}px</p>
                        <input type="range" min="1" max="10" value={drawSize}
                          onChange={e => setDrawSize(parseInt(e.target.value))}
                          className="w-full accent-primary h-1 cursor-pointer" />
                      </div>
                    )}
                    {/* Text default options panel — applies to next text box */}
                    {editorDisplayUrl && activeTool === "text" && (
                      <div className="absolute top-3 left-3 z-30 bg-background/95 backdrop-blur-sm border rounded-xl p-2.5 shadow-lg space-y-2 w-[170px]">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Text Color</p>
                        <div className="flex flex-wrap gap-1.5">
                          {penColors.map(c => (
                            <button
                              key={c.id}
                              onClick={() => setTextColor(c.value)}
                              className={cn("w-6 h-6 rounded-full border-2 transition-all", textColor === c.value ? "border-foreground scale-110 shadow-sm" : "border-transparent hover:scale-105")}
                              style={{ backgroundColor: c.value }}
                              title={c.label}
                            />
                          ))}
                        </div>
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider pt-1">Font · {textFontSize}px</p>
                        <input type="range" min="10" max="48" value={textFontSize}
                          onChange={e => setTextFontSize(parseInt(e.target.value))}
                          className="w-full accent-primary h-1 cursor-pointer" />
                        <p className="text-[9px] text-muted-foreground italic">Click on the scan to place a text box</p>
                      </div>
                    )}
                    {/* Ruler (measure) options panel */}
                    {editorDisplayUrl && activeTool === "measure" && (
                      <div className="absolute top-3 left-3 z-30 bg-background/95 backdrop-blur-sm border rounded-xl p-2.5 shadow-lg space-y-2 w-[170px]">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Ruler Color</p>
                        <div className="flex flex-wrap gap-1.5">
                          {penColors.map(c => (
                            <button
                              key={c.id}
                              onClick={() => setMeasureColor(c.value)}
                              className={cn("w-6 h-6 rounded-full border-2 transition-all", measureColor === c.value ? "border-foreground scale-110 shadow-sm" : "border-transparent hover:scale-105")}
                              style={{ backgroundColor: c.value }}
                              title={c.label}
                            />
                          ))}
                        </div>
                        <p className="text-[9px] text-muted-foreground italic">Click two points on the scan to measure</p>
                      </div>
                    )}
                    {/* Annotate (marker) options panel */}
                    {editorDisplayUrl && activeTool === "annotate" && (
                      <div className="absolute top-3 left-3 z-30 bg-background/95 backdrop-blur-sm border rounded-xl p-2.5 shadow-lg space-y-2 w-[170px]">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Marker Color</p>
                        <div className="flex flex-wrap gap-1.5">
                          {penColors.map(c => (
                            <button
                              key={c.id}
                              onClick={() => setAnnotateColor(c.value)}
                              className={cn("w-6 h-6 rounded-full border-2 transition-all", annotateColor === c.value ? "border-foreground scale-110 shadow-sm" : "border-transparent hover:scale-105")}
                              style={{ backgroundColor: c.value }}
                              title={c.label}
                            />
                          ))}
                        </div>
                        <p className="text-[9px] text-muted-foreground italic">Click on the scan to place a marker</p>
                      </div>
                    )}
                    <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between z-10 pointer-events-none">
                      <span className="text-[10px] text-white/70 bg-black/40 px-2 py-0.5 rounded truncate">
                        {activeModality === "mri" ? `Raw · ${uploadedFileName}` : uploadedFileName}
                      </span>
                      <span className="text-[10px] text-white/70 bg-black/40 px-2 py-0.5 rounded">{selectedView}</span>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="processing" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                    className="absolute inset-0 flex flex-col items-center justify-center p-4"
                  >
                    {(uploadedImageUrl || (activeModality === "mri" ? mriInputDisplayUrl : mriPreviewDisplayUrl)) && (
                      <div className="absolute inset-0 opacity-20 pointer-events-none">
                        <img
                          src={uploadedImageUrl ?? (activeModality === "mri" ? mriInputDisplayUrl : mriPreviewDisplayUrl) ?? undefined}
                          alt=""
                          className="w-full h-full object-contain"
                        />
                      </div>
                    )}
                    <div className="relative w-72 sm:w-80 p-5 rounded-xl bg-background border shadow-sm">
                    <div className="flex items-center gap-2 mb-2">
                      <Loader2 className="w-4 h-4 text-primary animate-spin" />
                      <p className="text-sm font-medium">Processing {activeModality === "xray" ? "X-Ray" : "MRI"}</p>
                    </div>
                    <p className="text-xs text-muted-foreground truncate mb-3">{uploadedFileName}</p>
                    <p className="text-xs text-foreground/90 leading-relaxed">
                      {activeModality === "xray"
                        ? stages.find((_, i) => i === currentStageIndex)?.label ?? "Running inference…"
                        : stages.find((_, i) => i === currentStageIndex)?.label ?? "Processing…"}
                    </p>
                    {isProcessing && uploadProgress < 100 && (
                      <div className="mt-3 space-y-1">
                        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                          <span>Uploading…</span>
                          <span className="text-mono">{Math.round(uploadProgress)}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                          <motion.div className="h-full bg-primary rounded-full" style={{ width: `${Math.min(uploadProgress, 100)}%` }} />
                        </div>
                      </div>
                    )}
                    {isProcessing && uploadProgress >= 100 && (
                      <p className="text-[10px] text-muted-foreground mt-2">Running model inference…</p>
                    )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Image controls bar */}
            <div className="h-10 border-t flex items-center gap-3 px-4 bg-muted/20 flex-shrink-0">
              <div className="flex items-center gap-2">
                <Sun className="w-3.5 h-3.5 text-muted-foreground" />
                <input type="range" min="50" max="150" value={brightness} onChange={e => setBrightness(parseInt(e.target.value))} className="w-16 accent-primary h-1" />
              </div>
              <div className="flex items-center gap-2">
                <Contrast className="w-3.5 h-3.5 text-muted-foreground" />
                <input type="range" min="50" max="150" value={contrast} onChange={e => setContrast(parseInt(e.target.value))} className="w-16 accent-primary h-1" />
              </div>
              <button className="ml-auto p-1 rounded hover:bg-muted transition-colors" title="Fullscreen">
                <Maximize2 className="w-3.5 h-3.5 text-muted-foreground" />
              </button>
            </div>
          </div>

          {/* AI Analysis panel */}
          <div className="flex-1 flex flex-col border-b">
            <div className="h-10 border-b flex items-center justify-between px-4 flex-shrink-0 bg-muted/20">
              <div className="flex items-center gap-2">
                <span className="section-header text-[10px]">AI Analysis</span>
                {diagnosticStage === "complete" && activeModality === "mri" && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Artifact-Free</span>
                )}
              </div>
            </div>

            {/* Grad-CAM area */}
            <div className="aspect-square max-h-[500px] bg-foreground/[0.02] flex items-center justify-center relative">
              <AnimatePresence mode="wait">
                {diagnosticStage === "complete" ? (
                  <motion.div key="gradcam" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full h-full p-3 flex items-center justify-center">
                    {activeModality === "xray" && primaryPreviewView ? (
                      <ScanImageTile
                        className="w-full max-w-md"
                        maxHeight="100%"
                        imageUrl={
                          showGradCAM && primaryPreviewView.gradcamUrl
                            ? primaryPreviewView.gradcamUrl
                            : uploadedImageUrl
                        }
                        label={
                          showGradCAM && primaryPreviewView.gradcamUrl
                            ? primaryPreviewView.name
                            : uploadedFileName || "Input scan"
                        }
                        sublabel={
                          showGradCAM && primaryPreviewView.gradcamUrl
                            ? `Grade ${primaryPreviewView.grade} · ${primaryPreviewView.confidence.toFixed(1)}%`
                            : `${selectedView} · compare models below`
                        }
                      />
                    ) : activeModality === "mri" && mriPreviewDisplayUrl ? (
                      <ScanImageTile
                        className="w-full max-w-md"
                        maxHeight="100%"
                        imageUrl={mriPreviewDisplayUrl}
                        label={mriViewModeLabel(mriViewMode)}
                        sublabel={
                          mriApiData
                            ? `${formatVolumeMeta(mriApiData)} · z=${mriActiveSlice?.slice_idx ?? mriApiData.primary_slice_idx ?? "—"}`
                            : selectedView
                        }
                      />
                    ) : (
                      <ScanImageTile
                        className="w-full max-w-md"
                        maxHeight="100%"
                        imageUrl={uploadedImageUrl}
                        label={uploadedFileName || "Scan"}
                        sublabel={selectedView}
                      />
                    )}
                  </motion.div>
                ) : diagnosticStage === "ready" ? (
                  <motion.div key="ready-gradcam" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="w-64 h-64 sm:w-72 sm:h-72 rounded-xl bg-foreground/5 border border-dashed flex flex-col items-center justify-center gap-3 px-6 text-center"
                  >
                    <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center">
                      <CheckCircle2 className="w-7 h-7 text-primary/60" />
                    </div>
                    <p className="text-sm font-medium">Input ready</p>
                    <p className="text-xs text-muted-foreground">Review the uploaded scan, then confirm input to run AI diagnosis.</p>
                  </motion.div>
                ) : isProcessing ? (
                  <motion.div key="waiting" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full h-full p-3 flex items-center justify-center">
                    <div className="w-full max-w-xs text-center space-y-3">
                      <Loader2 className="w-8 h-8 text-primary animate-spin mx-auto" />
                      <p className="text-sm font-medium">
                        {activeModality === "xray" ? "Evaluating all X-ray models…" : "Processing MRI…"}
                      </p>
                      {uploadedFileName && (
                        <p className="text-[10px] text-muted-foreground truncate">{uploadedFileName}</p>
                      )}
                      {isProcessing && uploadProgress < 100 && (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                            <span>Uploading…</span>
                            <span className="text-mono">{Math.round(uploadProgress)}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                            <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${uploadProgress}%` }} />
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="w-64 h-64 sm:w-72 sm:h-72 rounded-xl bg-foreground/5 border border-dashed flex flex-col items-center justify-center gap-3"
                  >
                    <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center">
                      <Layers className="w-7 h-7 text-primary/50" />
                    </div>
                    <p className="text-sm text-muted-foreground">Grad-CAM heatmap</p>
                    <p className="text-xs text-muted-foreground/60">Upload a scan to begin</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {diagnosticStage === "complete" && (
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-background/90 backdrop-blur-sm rounded-full p-1 shadow-sm border flex gap-1">
                  {activeModality === "mri" ? (
                    (["raw", "cleaned", "artifact", "gradcam"] as MriViewMode[]).map(mode => (
                      <button
                        key={mode}
                        onClick={() => setMriViewMode(mode)}
                        className={cn(
                          "px-2.5 py-1 rounded-full text-[10px] font-medium transition-all",
                          mriViewMode === mode ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
                        )}
                      >
                        {mriViewModeLabel(mode)}
                      </button>
                    ))
                  ) : (
                    <button onClick={() => setShowGradCAM(!showGradCAM)} className={cn("px-3 py-1 rounded-full text-[10px] font-medium transition-all", showGradCAM ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>
                      {showGradCAM ? "Hide" : "Show"} Heatmap
                    </button>
                  )}
                </div>
              )}
            </div>

          </div>
        </div>

        {/* MRI Pipeline details */}
        {activeModality === "mri" && diagnosticStage === "complete" && currentScan && <MriPipelinePanel scan={currentScan} />}

        {/* Results section */}
        <div className="px-4 sm:px-5 py-4 border-t">
          {diagnosticStage === "complete" ? (
            <div className="max-w-4xl mx-auto space-y-4">
              {!feedbackConfirmed && (
                <div className="p-3 rounded-xl border border-primary/20 bg-primary/5 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-foreground/90 leading-relaxed">
                    Review the AI findings for this {activeModality === "xray" ? "X-Ray" : "MRI"} scan.
                    Click <span className="font-medium">Agree</span> to update the patient&apos;s report, or <span className="font-medium">Override</span> if you disagree with the grade.
                  </p>
                </div>
              )}
              {feedbackConfirmed && (
                <div className="p-3 rounded-xl border border-success/20 bg-success/5 flex items-center justify-between gap-3 flex-wrap">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-success" />
                    <p className="text-xs font-medium text-success">
                      Report v{patient.report?.version ?? 1} updated — feedback confirmed
                    </p>
                  </div>
                  <button
                    onClick={() => navigate(`/reports/${patientId}`)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-success text-success-foreground text-xs font-medium hover:bg-success/90 transition-colors"
                  >
                    View report <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
              {/* Findings */}
              <div className="p-4 rounded-xl bg-success/5 border border-success/20">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle2 className="w-4 h-4 text-success" />
                  <p className="text-sm font-medium text-success">Analysis Complete</p>
                  <div className="ml-auto flex items-center gap-2">
                    <GradeBadge grade={result.grade} />
                    <ConfidenceGauge value={result.confidence} />
                  </div>
                </div>
                <div className="space-y-1.5">
                  {result.findings.map((f, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="w-1.5 h-1.5 rounded-full bg-success mt-1.5 flex-shrink-0" />
                      {f}
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-model performance breakdown */}
              <div className="p-4 rounded-xl border bg-card">
                <div className="flex items-center gap-2 mb-3">
                  <Brain className="w-4 h-4 text-primary" />
                  <p className="text-sm font-medium">Model Performance</p>
                  <span className="text-[10px] text-muted-foreground ml-1">
                    {activeModality === "xray" ? "X-Ray ensemble" : "MRI · MACS-Net + DeiT-S"}
                  </span>
                </div>
                <div className="overflow-hidden rounded-lg border">
                  <div className="grid grid-cols-12 gap-2 px-3 py-2 bg-muted/50 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                    <div className="col-span-6">Model</div>
                    <div className="col-span-2 text-center">Grade</div>
                    <div className="col-span-4">Confidence</div>
                  </div>
                  {modelRows.map((m) => {
                    const isEnsemble = activeModality === "xray" && (m.isEnsemble ?? m.id === "ensemble");
                    const isPrimary = activeModality === "mri" && m.isPrimary;
                    const isSelected =
                      activeModality === "xray"
                        ? selectedModelIds.has(m.id)
                        : selectedMriStageIds.has(m.id);
                    const rowClickable = diagnosticStage === "complete" && (activeModality === "xray" || activeModality === "mri");
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => {
                          if (activeModality === "xray") toggleModelSelection(m.id);
                          if (activeModality === "mri") toggleMriStageSelection(m.id);
                        }}
                        onDoubleClick={() => {
                          if (activeModality === "xray") selectOnlyModel(m.id);
                          if (activeModality === "mri") selectOnlyMriStage(m.id);
                        }}
                        className={cn(
                          "w-full grid grid-cols-12 gap-2 px-3 py-2 items-center text-xs border-t text-left transition-colors",
                          (isEnsemble || isPrimary) && "bg-primary/5",
                          rowClickable && "hover:bg-muted/50 cursor-pointer",
                          isSelected && "ring-1 ring-inset ring-primary/30 bg-primary/5",
                        )}
                      >
                        <div className="col-span-6 flex items-center gap-2 min-w-0">
                          <span className="font-medium truncate">{m.name}</span>
                          {isEnsemble && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Ensemble</span>
                          )}
                          {isPrimary && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Classifier</span>
                          )}
                          {isSelected && rowClickable && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Visible</span>
                          )}
                        </div>
                        <div className="col-span-2 flex justify-center">
                          {m.gradeDisplay ? (
                            <span className="text-[10px] font-medium text-muted-foreground">{m.gradeDisplay}</span>
                          ) : (
                            <GradeBadge grade={m.grade} />
                          )}
                        </div>
                        <div className="col-span-4">
                          {m.confidenceDisplay ? (
                            <span className="text-[10px] text-muted-foreground truncate block">{m.confidenceDisplay}</span>
                          ) : (
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                                <div className="h-full bg-primary rounded-full" style={{ width: `${m.confidence}%` }} />
                              </div>
                              <span className="text-mono text-[10px] text-muted-foreground w-10 text-right">{m.confidence.toFixed(1)}%</span>
                            </div>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
                {activeModality === "xray" && xrayApiData && (
                  <>
                    <p className="text-[10px] text-muted-foreground mt-2">
                      {xrayApiData.model_count ?? Object.keys(xrayApiData.individual_results).length} models evaluated.
                      Click a row to toggle; double-click to view only that model.
                    </p>
                    <XrayModelEvaluationPanel
                      data={xrayApiData}
                      selectedIds={selectedModelIds}
                      showHeatmap={showGradCAM}
                      baseImageUrl={uploadedImageUrl}
                      inputFileName={uploadedFileName}
                    />
                  </>
                )}
                {activeModality === "mri" && mriApiData && (
                  <>
                    <p className="text-[10px] text-muted-foreground mt-2">
                      2-stage pipeline (MACS-Net → DeiT-S). Click a row to toggle; double-click to view only that stage.
                    </p>
                    <MriModelEvaluationPanel
                      data={mriApiData}
                      selectedIds={selectedMriStageIds}
                      viewMode={mriViewMode}
                      activeSliceIdx={mriGallerySliceIdx}
                      onSliceChange={setMriGallerySliceIdx}
                      inputFileName={uploadedFileName}
                    />
                    {mriApiData.ground_truth_labels && mriApiData.ground_truth_labels.length > 0 && (
                      <p className="text-[10px] text-muted-foreground mt-2">
                        SKM-TEA reference: {mriApiData.ground_truth_labels.join(", ")}
                      </p>
                    )}
                    {mriApiData.multilabel_predictions.filter(l => l.predicted).length > 0 && (
                      <div className="mt-3 rounded-lg border overflow-hidden">
                        <div className="px-3 py-2 bg-muted/50 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                          SKM-TEA categories (DeiT-S)
                        </div>
                        {mriApiData.multilabel_predictions
                          .filter(l => l.predicted)
                          .sort((a, b) => b.probability - a.probability)
                          .map((l) => (
                            <div key={l.name} className="flex items-center justify-between px-3 py-1.5 text-xs border-t">
                              <span className="truncate pr-2">{l.name}</span>
                              <span className="text-mono text-muted-foreground flex-shrink-0">{l.probability.toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Clinical Interpretation + References */}
              <ClinicalInterpretation patient={patient} analysis={computeJointAnalysis(patient, initialBatchResult)} batchResult={initialBatchResult} />

              {/* Actions */}
              <div className="flex items-center justify-between flex-wrap gap-3">
                <p className="text-sm">
                  <span className="text-muted-foreground">Classification: </span>
                  <span className="font-medium">Grade {result.grade} Osteoarthritis</span>
                  <span className="text-muted-foreground ml-1">({result.confidence}%)</span>
                </p>
                <div className="flex items-center gap-2">
                  <button onClick={() => editorRef.current?.exportPNG(`${patient.name.replace(/\s+/g, "_")}_annotated.png`)} className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm font-medium text-primary hover:bg-primary hover:text-primary-foreground transition-colors">
                    <Download className="w-4 h-4" />Download
                  </button>
                  <button className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm font-medium text-muted-foreground hover:bg-muted transition-colors">
                    <Save className="w-4 h-4" />Save to Profile
                  </button>
                  <button
                    onClick={handleAgree}
                    disabled={feedbackConfirmed}
                    className={cn(
                      "inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm font-medium transition-colors",
                      feedbackConfirmed
                        ? "text-muted-foreground bg-muted cursor-not-allowed"
                        : "text-success hover:bg-success hover:text-success-foreground"
                    )}
                  >
                    <Check className="w-4 h-4" />{feedbackConfirmed ? "Agreed" : "Agree"}
                  </button>
                  <button
                    onClick={() => { setShowOverridePanel(!showOverridePanel); if (overrideGrade == null) setOverrideGrade(result.grade); }}
                    disabled={feedbackConfirmed}
                    className={cn(
                      "inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm font-medium transition-colors",
                      feedbackConfirmed ? "text-muted-foreground cursor-not-allowed" : "text-warning hover:bg-warning hover:text-warning-foreground"
                    )}
                  >
                    <X className="w-4 h-4" />Override
                  </button>
                </div>
              </div>

              {/* Override panel */}
              <AnimatePresence>
                {showOverridePanel && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                    className="p-4 rounded-xl bg-warning/5 border border-warning/20 overflow-hidden"
                  >
                    <p className="text-xs font-medium mb-3">Manual Grade Override</p>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-xs text-muted-foreground">Grade:</span>
                      {[0, 1, 2, 3, 4].map(g => (
                        <button key={g} onClick={() => setOverrideGrade(g)} className={cn("w-8 h-8 rounded-lg text-xs font-medium transition-all", overrideGrade === g ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:text-foreground")}>{g}</button>
                      ))}
                    </div>
                    <textarea value={overrideNotes} onChange={e => setOverrideNotes(e.target.value)} placeholder="Clinical reasoning for override..." className="w-full px-3 py-2 rounded-lg border bg-background text-sm resize-none h-16 focus:outline-none focus:ring-2 focus:ring-ring/20" />
                    <div className="flex justify-end mt-2 gap-2">
                      <button onClick={() => setShowOverridePanel(false)} className="px-3 py-1.5 text-xs rounded-lg border hover:bg-muted transition-colors">Cancel</button>
                      <button onClick={handleOverrideSubmit} className="px-3 py-1.5 text-xs rounded-lg bg-warning text-warning-foreground hover:bg-warning/90 transition-colors">Submit Override</button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ) : diagnosticStage === "ready" ? (
            <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 py-2">
              <div>
                <p className="text-sm font-medium">Scan uploaded — confirm before analysis</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {uploadedFileName} · {activeModality === "xray" ? "X-Ray" : "MRI"} · {selectedView}
                  {requiredMods.length > 1 && !allRequiredReady && (
                    <span className="block mt-1 text-warning">Upload {requiredMods.filter(m => !getSessionForMod(m).fileName).map(m => m === "xray" ? "X-Ray" : "MRI").join(" and ")} to continue</span>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleFileSelect}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors"
                >
                  <Upload className="w-4 h-4" />Change file
                </button>
                <button
                  onClick={handleStartAnalysis}
                  disabled={!activeModReady}
                  className={cn(
                    "inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                    activeModReady
                      ? "bg-primary text-primary-foreground hover:bg-primary/90"
                      : "bg-muted text-muted-foreground cursor-not-allowed"
                  )}
                >
                  <Check className="w-4 h-4" />Review &amp; confirm input
                </button>
              </div>
            </div>
          ) : diagnosticStage === "idle" ? (
            <div className="max-w-4xl mx-auto py-3 px-4">
              <div className="rounded-lg border border-dashed bg-muted/15 px-4 py-3 flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Upload className="w-4 h-4 text-primary" />
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Upload a scan in the viewer above, then confirm input to run AI diagnosis. Results and clinical interpretation will appear here when analysis completes.
                </p>
              </div>
            </div>
          ) : isProcessing ? (
            <div className="max-w-4xl mx-auto py-6 px-4">
              <div className="rounded-xl border bg-card p-6 flex flex-col items-center gap-4 text-center">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
                <div>
                  <p className="text-sm font-medium">
                    Analyzing {activeModality === "xray" ? "X-Ray" : "MRI"}…
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stages.find((_, i) => i === currentStageIndex)?.label ?? "Processing…"}
                  </p>
                </div>
                {uploadProgress < 100 && (
                  <div className="w-full max-w-xs space-y-1">
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>Uploading…</span>
                      <span className="text-mono">{Math.round(uploadProgress)}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${uploadProgress}%` }} />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <ConfirmInputDialog
        open={showInputConfirm}
        onClose={() => setShowInputConfirm(false)}
        onConfirm={runAnalysis}
        modality={activeModality}
        view={selectedView}
        fileName={uploadedFileName}
        previewUrl={uploadedImageUrl}
        patientName={patient.name}
        uploads={confirmUploads}
        mriServerSample={mriUsesServerSample && activeModality === "mri"}
      />
    </div>
  );
}

// ============================================================
// Main Page — phase state machine
// ============================================================
// ============================================================
// History view — past diagnoses with inputs and outputs
// ============================================================
function HistoryView({ onOpen, onBack }: { onOpen: (p: Patient) => void; onBack: () => void }) {
  const { patients } = usePatients();
  const [search, setSearch] = useState("");
  const history = useMemo(() => {
    const items = patients
      .filter(p => p.status !== "pending" && p.scans.length > 0)
      .flatMap(p => p.scans.map(s => ({ patient: p, scan: s })))
      .filter(({ scan }) => scan.grade != null)
      .sort((a, b) => b.scan.date.localeCompare(a.scan.date));
    if (!search) return items;
    const q = search.toLowerCase();
    return items.filter(({ patient }) =>
      patient.name.toLowerCase().includes(q) || patient.id.toLowerCase().includes(q),
    );
  }, [search]);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 overflow-auto">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
        <div className="flex items-center justify-between gap-3 mb-3">
          <div>
            <button onClick={onBack} className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1 mb-1.5">
              <ArrowLeft className="w-3 h-3" /> Back
            </button>
            <h1 className="text-lg sm:text-xl font-semibold tracking-tight">Diagnosis History</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Past AI analyses with inputs and outputs.</p>
          </div>
          <span className="text-[11px] text-muted-foreground shrink-0">{history.length} record{history.length !== 1 ? "s" : ""}</span>
        </div>

        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search by name or patient ID..."
            className="w-full pl-9 pr-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/30"
          />
        </div>

        <div className="rounded-lg border bg-card divide-y max-h-[min(58vh,580px)] overflow-y-auto">
          {history.map(({ patient, scan }) => (
            <div key={`${patient.id}-${scan.id}`} className="px-3 py-2.5 grid grid-cols-1 sm:grid-cols-[1fr_auto] sm:items-center gap-2 hover:bg-muted/30 transition-colors">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <p className="text-sm font-medium truncate">{patient.name}</p>
                  <span className="text-[10px] font-mono text-muted-foreground">{patient.id}</span>
                </div>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span className="inline-flex items-center gap-1">
                    <Calendar className="w-3 h-3" />{scan.date}
                  </span>
                  <span className="uppercase font-medium tracking-wide">{scan.modality === "xray" ? "X-Ray" : "MRI"}{scan.view ? ` · ${scan.view}` : ""}</span>
                  <span className="truncate">{scan.region}</span>
                </div>
                <p className="text-[10px] text-muted-foreground mt-1 truncate">
                  Input: {scan.preprocessing.join(" → ")} · Model: {scan.modelUsed}
                </p>
              </div>
              <div className="flex items-center gap-3 sm:gap-4 justify-between sm:justify-end">
                <div className="flex items-center gap-3">
                  <div className="text-center">
                    <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Grade</p>
                    <div className="mt-0.5"><GradeBadge grade={scan.grade as number} /></div>
                  </div>
                  <div className="text-center">
                    <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Conf.</p>
                    <p className="text-xs font-semibold tabular-nums">{scan.aiConfidence?.toFixed(1)}%</p>
                  </div>
                </div>
                <button
                  onClick={() => onOpen(patient)}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] font-medium border hover:bg-muted transition-colors"
                >
                  Open <ChevronRight className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
          {history.length === 0 && (
            <div className="py-16 flex flex-col items-center gap-2 text-muted-foreground">
              <Clock className="w-8 h-8" />
              <p className="text-sm">No diagnosis history yet</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

type Phase = "select" | "history" | "inputs" | "confirm" | "processing" | "results" | "workspace";

export default function DiagnosticsPage() {
  const { patients, getPatient, applyAnalysisResult } = usePatients();
  const [searchParams, setSearchParams] = useSearchParams();
  const preselectedId = searchParams.get("patient");
  const [phase, setPhase] = useState<Phase>(preselectedId ? "workspace" : "select");
  const [cohortIds, setCohortIds] = useState<string[]>([]);
  const [cohortInputs, setCohortInputs] = useState<Map<string, CohortInputEntry>>(new Map());
  const [cohortFiles, setCohortFiles] = useState<CohortFilesMap>(new Map());
  const [cohortResults, setCohortResults] = useState<Map<string, BatchPatientResult>>(new Map());
  const [workspacePatientId, setWorkspacePatientId] = useState<string | null>(preselectedId);

  const cohort = useMemo(
    () => cohortIds.map(id => getPatient(id)).filter((p): p is Patient => !!p),
    [cohortIds, patients, getPatient],
  );

  const goSelect = () => {
    setPhase("select");
    setCohortIds([]);
    setCohortInputs(new Map());
    setCohortFiles(new Map());
    setCohortResults(new Map());
    setWorkspacePatientId(null);
    setSearchParams({});
  };

  const openWorkspace = (p: Patient) => {
    setWorkspacePatientId(p.id);
    setPhase("workspace");
    setSearchParams({ patient: p.id });
  };

  const handlePatientConfirm = (selected: Patient[]) => {
    if (selected.length === 1) {
      openWorkspace(selected[0]);
      return;
    }
    setCohortIds(selected.map(p => p.id));
    setCohortInputs(new Map());
    setCohortFiles(new Map());
    setCohortResults(new Map());
    setPhase("inputs");
  };

  return (
    <motion.div initial={false} animate={{ opacity: 1 }} transition={{ duration: 0.25 }} className="flex-1 flex flex-col min-h-0 w-full">
      <AnimatePresence mode="wait">
        {phase === "select" && (
          <PatientSelector key="selector" onConfirm={handlePatientConfirm} onOpenHistory={() => setPhase("history")} />
        )}
        {phase === "history" && (
          <HistoryView key="history" onBack={() => setPhase("select")} onOpen={openWorkspace} />
        )}
        {phase === "inputs" && (
          <BatchInputScreen
            key="inputs"
            patients={cohort}
            onCancel={() => setPhase("select")}
            onContinue={(inputs, files) => { setCohortInputs(inputs); setCohortFiles(files); setPhase("confirm"); }}
          />
        )}
        {phase === "confirm" && (
          <ConfirmationScreen
            key="confirm"
            patients={cohort}
            cohortInputs={cohortInputs}
            onCancel={() => setPhase("inputs")}
            onStart={() => setPhase("processing")}
          />
        )}
        {phase === "processing" && (
          <ProcessingScreen
            key="processing"
            patients={cohort}
            cohortInputs={cohortInputs}
            cohortFiles={cohortFiles}
            applyAnalysisResult={applyAnalysisResult}
            onComplete={(results) => { setCohortResults(results); setPhase("results"); }}
            onCancel={goSelect}
          />
        )}
        {phase === "results" && (
          <ResultsOverview
            key="results"
            patients={cohort}
            cohortInputs={cohortInputs}
            cohortResults={cohortResults}
            onOpenWorkspace={openWorkspace}
            onBackToSelect={goSelect}
          />
        )}
        {phase === "workspace" && workspacePatientId && (
          <DiagnosticWorkspace
            key={`workspace-${workspacePatientId}`}
            patientId={workspacePatientId}
            initialBatchResult={cohortResults.get(workspacePatientId)}
            cohortInputEntry={cohortInputs.get(workspacePatientId)}
            cohortFiles={cohortFiles.get(workspacePatientId)}
            onBack={() => {
              if (cohortIds.length > 0) { setPhase("results"); setSearchParams({}); }
              else goSelect();
            }}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
