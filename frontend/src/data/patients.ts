export type Modality = "xray" | "mri";

export interface ScanEntry {
  id: string;
  modality: Modality;
  date: string;
  view?: string; // AP, Lateral, Sagittal, Coronal, Axial
  region: string;
  grade: number | null;
  aiConfidence: number | null;
  modelUsed: string;
  artifactRemoval?: {
    applied: boolean;
    method: string; // "Swin-UNet" | "none"
    dataset: string; // "KMAR-50K" | "N/A"
    qualityScore: number; // 0-100 post-enhancement quality
  };
  preprocessing: string[];
}

export interface TimelineEntry {
  date: string;
  type: "scan" | "diagnosis" | "note" | "report";
  summary: string;
  grade?: number;
  confidence?: number;
}

export interface ReportModelSnapshot {
  modelId: string;
  displayName: string;
  grade: number;
  confidence: number;
  gradcamDataUrl: string | null;
}

/** Per-modality snapshot stored after batch or multi-modality analysis. */
export interface ModalityReportSnapshot {
  modality: Modality;
  grade: number;
  confidence: number;
  findings: string[];
  modelUsed: string;
  inputFileName: string;
  view: string;
  inputImageDataUrl?: string | null;
  ensembleGradcamDataUrl?: string | null;
  modelResults?: ReportModelSnapshot[];
}

/** One living report per patient — updated when the doctor confirms AI feedback. */
export interface PatientReport {
  aiGrade: number;
  aiConfidence: number;
  finalGrade: number;
  modality: Modality;
  view: string;
  region: string;
  inputFileName: string;
  findings: string[];
  diagnosisSummary: string;
  modelUsed: string;
  inputImageDataUrl?: string | null;
  ensembleGradcamDataUrl?: string | null;
  modelResults?: ReportModelSnapshot[];
  /** Full per-modality outputs when batch or joint analysis ran. */
  modalitySnapshots?: ModalityReportSnapshot[];
  doctorConfirmed: boolean;
  doctorOverride: boolean;
  overrideNotes?: string;
  updatedAt: string;
  version: number;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: "Male" | "Female";
  bmi: number;
  history: string;
  symptoms: string;
  painLevel: number;
  grade: number | null;
  aiConfidence: number | null;
  lastVisit: string;
  status: "pending" | "analyzed" | "confirmed" | "flagged";
  modality: Modality;
  scans: ScanEntry[];
  timeline: TimelineEntry[];
  report?: PatientReport | null;
}

export const mockPatients: Patient[] = [
  {
    id: "PT-8842",
    name: "Nguyễn Văn An",
    age: 65,
    gender: "Male",
    bmi: 27.3,
    history: "Chronic bilateral knee pain, 3 years. Previous meniscus repair (2019).",
    symptoms: "Morning stiffness >30min, crepitus on flexion, reduced ROM.",
    painLevel: 7,
    grade: 3,
    aiConfidence: 94.2,
    lastVisit: "2026-03-15",
    status: "confirmed",
    modality: "xray",
    scans: [
      {
        id: "SCN-8842-01", modality: "xray", date: "2026-03-15", view: "AP", region: "Bilateral Knee",
        grade: 3, aiConfidence: 94.2, modelUsed: "DenseNet201 (Ensemble)",
        preprocessing: ["CLAHE", "Denoise", "Normalization"],
      },
      {
        id: "SCN-8842-02", modality: "xray", date: "2025-09-10", view: "AP", region: "Bilateral Knee",
        grade: 2, aiConfidence: 88.1, modelUsed: "DenseNet201 (Ensemble)",
        preprocessing: ["CLAHE", "Denoise"],
      },
      {
        id: "SCN-8842-03", modality: "mri", date: "2026-03-15", view: "Sagittal", region: "Bilateral Knee",
        grade: 3, aiConfidence: 91.4, modelUsed: "Swin-UNet + DEiT-S",
        artifactRemoval: { applied: true, method: "Swin-UNet", dataset: "KMAR-50K", qualityScore: 90 },
        preprocessing: ["Artifact Removal (Swin-UNet)", "CLAHE", "Normalization"],
      },
    ],
    timeline: [
      { date: "2026-03-15", type: "scan", summary: "Bilateral AP knee X-ray uploaded" },
      { date: "2026-03-15", type: "diagnosis", summary: "AI Classification: Grade 3 OA (94.2%)", grade: 3, confidence: 94.2 },
      { date: "2026-03-15", type: "note", summary: "Confirmed by Dr. Quốc Châu. Joint space narrowing noted." },
      { date: "2025-09-10", type: "scan", summary: "Follow-up AP knee X-ray" },
      { date: "2025-09-10", type: "diagnosis", summary: "AI Classification: Grade 2 OA (88.1%)", grade: 2, confidence: 88.1 },
    ],
  },
  {
    id: "PT-7291",
    name: "Trần Thị Mai",
    age: 52,
    gender: "Female",
    bmi: 24.1,
    history: "No prior knee conditions. Family history of OA.",
    symptoms: "Intermittent pain after prolonged walking, mild swelling.",
    painLevel: 4,
    grade: 2,
    aiConfidence: 87.6,
    lastVisit: "2026-03-14",
    status: "analyzed",
    modality: "mri",
    scans: [
      {
        id: "SCN-7291-01", modality: "mri", date: "2026-03-14", view: "Sagittal", region: "Right Knee",
        grade: 2, aiConfidence: 87.6, modelUsed: "Swin-UNet + DenseNet201",
        artifactRemoval: { applied: true, method: "Swin-UNet", dataset: "KMAR-50K", qualityScore: 92 },
        preprocessing: ["Artifact Removal (Swin-UNet)", "CLAHE", "Normalization"],
      },
      {
        id: "SCN-7291-02", modality: "mri", date: "2026-03-14", view: "Coronal", region: "Right Knee",
        grade: 2, aiConfidence: 85.3, modelUsed: "Swin-UNet + ViT-B/16",
        artifactRemoval: { applied: true, method: "Swin-UNet", dataset: "KMAR-50K", qualityScore: 89 },
        preprocessing: ["Artifact Removal (Swin-UNet)", "Denoise", "Normalization"],
      },
      {
        id: "SCN-7291-03", modality: "xray", date: "2026-03-14", view: "AP", region: "Right Knee",
        grade: 2, aiConfidence: 86.7, modelUsed: "DenseNet201 (Ensemble)",
        preprocessing: ["CLAHE", "Denoise", "Normalization"],
      },
    ],
    timeline: [
      { date: "2026-03-14", type: "scan", summary: "Right knee MRI (Sagittal + Coronal) uploaded" },
      { date: "2026-03-14", type: "diagnosis", summary: "AI Classification: Grade 2 OA (87.6%) — MRI artifact removal applied", grade: 2, confidence: 87.6 },
    ],
  },
  {
    id: "PT-6105",
    name: "Lê Hoàng Dũng",
    age: 71,
    gender: "Male",
    bmi: 30.2,
    history: "Bilateral knee OA diagnosed 2020. Total knee replacement (L) 2023.",
    symptoms: "Right knee: severe pain at rest, bone-on-bone sensation.",
    painLevel: 9,
    grade: 4,
    aiConfidence: 97.8,
    lastVisit: "2026-03-12",
    status: "confirmed",
    modality: "xray",
    scans: [
      {
        id: "SCN-6105-01", modality: "xray", date: "2026-03-12", view: "AP", region: "Right Knee",
        grade: 4, aiConfidence: 97.8, modelUsed: "DenseNet201 (Ensemble)",
        preprocessing: ["CLAHE", "Denoise", "Normalization"],
      },
      {
        id: "SCN-6105-02", modality: "xray", date: "2026-03-12", view: "Lateral", region: "Right Knee",
        grade: 4, aiConfidence: 96.5, modelUsed: "DenseNet201 (Ensemble)",
        preprocessing: ["CLAHE", "Denoise"],
      },
    ],
    timeline: [
      { date: "2026-03-12", type: "scan", summary: "Right knee AP & lateral X-ray uploaded" },
      { date: "2026-03-12", type: "diagnosis", summary: "AI Classification: Grade 4 OA (97.8%)", grade: 4, confidence: 97.8 },
      { date: "2026-03-12", type: "report", summary: "Clinical report generated. Referred to orthopedics." },
    ],
  },
  {
    id: "PT-5530",
    name: "Phạm Minh Châu",
    age: 45,
    gender: "Female",
    bmi: 22.8,
    history: "ACL reconstruction (2018). Active runner.",
    symptoms: "Mild discomfort after exercise, no swelling.",
    painLevel: 2,
    grade: 1,
    aiConfidence: 72.3,
    lastVisit: "2026-03-10",
    status: "flagged",
    modality: "mri",
    scans: [
      {
        id: "SCN-5530-01", modality: "mri", date: "2026-03-10", view: "Sagittal", region: "Bilateral Knee",
        grade: 1, aiConfidence: 72.3, modelUsed: "Swin-UNet + ViT-B/16",
        artifactRemoval: { applied: true, method: "Swin-UNet", dataset: "KMAR-50K", qualityScore: 78 },
        preprocessing: ["Artifact Removal (Swin-UNet)", "CLAHE", "Denoise"],
      },
      {
        id: "SCN-5530-02", modality: "mri", date: "2026-03-10", view: "Axial", region: "Bilateral Knee",
        grade: 1, aiConfidence: 68.9, modelUsed: "Swin-UNet + DenseNet201",
        artifactRemoval: { applied: true, method: "Swin-UNet", dataset: "KMAR-50K", qualityScore: 74 },
        preprocessing: ["Artifact Removal (Swin-UNet)", "Normalization"],
      },
    ],
    timeline: [
      { date: "2026-03-10", type: "scan", summary: "Bilateral knee MRI uploaded" },
      { date: "2026-03-10", type: "diagnosis", summary: "AI Classification: Grade 1 OA (72.3%) — MRI pipeline", grade: 1, confidence: 72.3 },
      { date: "2026-03-10", type: "note", summary: "Flagged: Low confidence. Doctor review pending." },
    ],
  },
  {
    id: "PT-4417",
    name: "Võ Thanh Sơn",
    age: 58,
    gender: "Male",
    bmi: 26.5,
    history: "Type 2 Diabetes. Hypertension. Sedentary lifestyle.",
    symptoms: "Bilateral knee pain, worse on stairs. Morning stiffness.",
    painLevel: 6,
    grade: null,
    aiConfidence: null,
    lastVisit: "2026-03-17",
    status: "pending",
    modality: "xray",
    scans: [
      {
        id: "SCN-4417-01", modality: "xray", date: "2026-03-17", view: "AP", region: "Bilateral Knee",
        grade: null, aiConfidence: null, modelUsed: "Pending",
        preprocessing: [],
      },
    ],
    timeline: [
      { date: "2026-03-17", type: "scan", summary: "Bilateral AP knee X-ray uploaded. Awaiting analysis." },
    ],
  },
  {
    id: "PT-9901",
    name: "Lý Thị Hoa",
    age: 48,
    gender: "Female",
    bmi: 24.1,
    history: "Right knee effusion for 6 months. No prior knee surgery.",
    symptoms: "Intermittent swelling after standing, occasional locking sensation.",
    painLevel: 5,
    grade: null,
    aiConfidence: null,
    lastVisit: "2026-05-31",
    status: "pending",
    modality: "mri",
    scans: [
      {
        id: "SCN-9901-01", modality: "mri", date: "2026-05-31", view: "Axial", region: "Right Knee",
        grade: null, aiConfidence: null, modelUsed: "Pending",
        preprocessing: [],
      },
    ],
    timeline: [
      { date: "2026-05-31", type: "scan", summary: "Right knee MRI ordered. Ready for MACS-Net + DeiT-S analysis." },
    ],
  },
  {
    id: "PT-9902",
    name: "Đặng Văn Khôi",
    age: 61,
    gender: "Male",
    bmi: 28.4,
    history: "Occupational knee load (construction). Family history of osteoarthritis.",
    symptoms: "Progressive medial knee pain, crepitus, reduced flexion.",
    painLevel: 6,
    grade: null,
    aiConfidence: null,
    lastVisit: "2026-05-31",
    status: "pending",
    modality: "xray",
    scans: [
      {
        id: "SCN-9902-01", modality: "xray", date: "2026-05-31", view: "AP", region: "Left Knee",
        grade: null, aiConfidence: null, modelUsed: "Pending",
        preprocessing: [],
      },
    ],
    timeline: [
      { date: "2026-05-31", type: "scan", summary: "Left knee AP X-ray uploaded. Awaiting ensemble analysis." },
    ],
  },
];
