import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, Download, Printer, Send, Edit3, TrendingUp, TrendingDown, Minus, Shield, Activity, Bone, Stethoscope, Calendar, ClipboardCheck, AlertCircle, X, Eye, FileText, Loader2 } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { usePatients } from "@/context/PatientContext";
import { GradeBadge } from "@/components/GradeBadge";
import { StatusBadge } from "@/components/StatusBadge";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts";
import { getReportDataURL, downloadReportPDF } from "@/lib/generateReportPDF";

const containerVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] as const } },
};

// KL grading criteria reference
const klCriteria = [
  { grade: 0, label: "Normal", description: "No radiographic features of OA" },
  { grade: 1, label: "Doubtful", description: "Doubtful narrowing of joint space; possible osteophytic lipping" },
  { grade: 2, label: "Minimal", description: "Definite osteophytes; possible narrowing of joint space" },
  { grade: 3, label: "Moderate", description: "Moderate osteophytes; definite narrowing; some sclerosis; possible deformity" },
  { grade: 4, label: "Severe", description: "Large osteophytes; marked narrowing; severe sclerosis; definite deformity" },
];

export default function ReportDetailPage() {
  const navigate = useNavigate();
  const { patientId } = useParams();
  const { getPatient } = usePatients();
  const patient = getPatient(patientId ?? "") ?? getPatient("PT-8842")!;
  const report = patient.report;

  const diagnosisHistory = patient.timeline
    .filter(e => e.type === "report" && e.grade !== undefined)
    .reverse()
    .slice(0, 5)
    .map(e => ({
      date: e.date,
      grade: e.grade!,
      confidence: e.confidence!,
      summary: e.summary,
    }));

  const progressionData = diagnosisHistory.length > 0
    ? diagnosisHistory.map(d => ({
        date: new Date(d.date).toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
        grade: d.grade,
        confidence: d.confidence,
      }))
    : report
      ? [{ date: new Date(report.updatedAt).toLocaleDateString("en-US", { month: "short", year: "2-digit" }), grade: report.finalGrade, confidence: report.aiConfidence }]
      : [];

  const gradeChanged = diagnosisHistory.length >= 2;
  const gradeTrend = gradeChanged
    ? diagnosisHistory[diagnosisHistory.length - 1].grade - diagnosisHistory[0].grade
    : 0;

  const displayGrade = report?.finalGrade ?? patient.grade;
  const displayConfidence = report?.aiConfidence ?? patient.aiConfidence;

  const riskData = [
    { factor: "Age", value: Math.min(100, (patient.age / 80) * 100) },
    { factor: "BMI", value: Math.min(100, (patient.bmi / 35) * 100) },
    { factor: "Pain", value: (patient.painLevel / 10) * 100 },
    { factor: "Grade", value: displayGrade !== null ? (displayGrade / 4) * 100 : 0 },
    { factor: "History", value: patient.history.length > 50 ? 75 : 35 },
  ];

  const riskScore = displayGrade !== null
    ? Math.round(((displayGrade / 4) * 40) + ((patient.painLevel / 10) * 25) + ((patient.bmi > 25 ? (patient.bmi - 25) / 10 : 0) * 20) + ((patient.age > 55 ? (patient.age - 55) / 25 : 0) * 15))
    : null;

  const riskLevel = riskScore !== null
    ? riskScore >= 70 ? "High" : riskScore >= 40 ? "Moderate" : "Low"
    : "N/A";

  const currentKL = klCriteria.find(k => k.grade === displayGrade);

  // PDF Preview state
  const [showPdfPreview, setShowPdfPreview] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfDataUrl, setPdfDataUrl] = useState<string>("");

  const handlePreviewPdf = () => {
    setPdfLoading(true);
    setShowPdfPreview(true);
    // Simulate slight delay for PDF generation
    setTimeout(() => {
      const dataUrl = getReportDataURL(patient);
      setPdfDataUrl(dataUrl);
      setPdfLoading(false);
    }, 800);
  };

  const handleDownloadPdf = () => {
    downloadReportPDF(patient);
  };

  const handlePrintReport = () => {
    const dataUrl = getReportDataURL(patient);
    const win = window.open(dataUrl, "_blank");
    if (win) {
      win.addEventListener("load", () => win.print());
    }
  };

  return (
    <>
    {/* PDF Preview Modal */}
    <AnimatePresence>
      {showPdfPreview && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={() => setShowPdfPreview(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-background rounded-xl shadow-2xl border w-[90vw] h-[90vh] max-w-5xl flex flex-col overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <div className="h-14 border-b flex items-center justify-between px-5 flex-shrink-0">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-primary" />
                <div>
                  <p className="text-sm font-medium">PDF Preview — {patient.name}</p>
                  <p className="text-[10px] text-muted-foreground">{patient.id} · KneeXpert Clinical Report</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownloadPdf}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Download PDF
                </button>
                <button
                  onClick={handlePrintReport}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors"
                >
                  <Printer className="w-4 h-4" />
                  Print
                </button>
                <button
                  onClick={() => setShowPdfPreview(false)}
                  className="p-2 rounded-lg hover:bg-muted transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="flex-1 bg-muted/30 p-4 overflow-auto flex justify-center">
              {pdfLoading ? (
                <div className="flex flex-col items-center justify-center gap-3">
                  <Loader2 className="w-8 h-8 text-primary animate-spin" />
                  <p className="text-sm text-muted-foreground">Generating PDF report...</p>
                </div>
              ) : (
                <iframe
                  src={pdfDataUrl}
                  className="w-full max-w-[800px] h-full rounded-lg border shadow-lg bg-white"
                  title="PDF Preview"
                />
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>

    <div className="h-screen flex overflow-hidden">
      <div className="flex-1 overflow-auto">
        <motion.div variants={containerVariants} initial="hidden" animate="visible" className="max-w-5xl mx-auto py-5 px-4 lg:px-5">
          <button onClick={() => navigate("/reports")} className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6">
            <ArrowLeft className="w-4 h-4" />
            Back to Reports
          </button>

          {/* Report Header */}
          <div className="card-clinical mb-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="section-header mb-2">KneeXpert Clinical Diagnostic Report</p>
                <h1 className="text-2xl font-medium tracking-tight">{patient.name}</h1>
                <p className="text-mono text-xs text-muted-foreground mt-1">
                  {patient.id}
                  {report
                    ? ` · Report v${report.version} · Updated ${report.updatedAt}`
                    : " · Awaiting confirmed diagnosis"}
                </p>
                {report && (
                  <p className="text-[11px] text-muted-foreground mt-1">
                    Input: {report.inputFileName} · {report.modality === "xray" ? "X-Ray" : "MRI"}{report.view ? ` · ${report.view}` : ""}
                    {report.doctorOverride && " · Doctor override applied"}
                  </p>
                )}
              </div>
              <StatusBadge status={patient.status} />
            </div>

            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 pt-4 border-t">
              {[
                { label: "Age", value: `${patient.age}` },
                { label: "Gender", value: patient.gender },
                { label: "BMI", value: `${patient.bmi}` },
                { label: "Pain Level", value: `${patient.painLevel}/10` },
                { label: "Last Visit", value: patient.lastVisit },
                { label: "Total Scans", value: `${patient.timeline.filter(e => e.type === "scan").length}` },
              ].map(item => (
                <div key={item.label}>
                  <p className="section-header mb-1">{item.label}</p>
                  <p className="text-sm font-medium">{item.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Risk Assessment */}
          <div className="card-clinical mb-6">
            <div className="flex items-center justify-between mb-3">
              <p className="section-header">Risk Assessment</p>
              {riskScore !== null && (
                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                  riskLevel === "High" ? "bg-destructive/10 text-destructive" :
                  riskLevel === "Moderate" ? "bg-warning/10 text-warning" :
                  "bg-success/10 text-success"
                }`}>
                  <Shield className="w-3 h-3" />
                  {riskLevel} Risk · Score: {riskScore}/100
                </div>
              )}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <ResponsiveContainer width="100%" height={200}>
                  <RadarChart data={riskData}>
                    <PolarGrid stroke="hsl(220, 13%, 91%)" />
                    <PolarAngleAxis dataKey="factor" tick={{ fontSize: 10 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar dataKey="value" stroke="hsl(217, 91%, 60%)" fill="hsl(217, 91%, 60%)" fillOpacity={0.15} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 flex flex-col justify-center">
                {[
                  { label: "Age Factor", detail: patient.age >= 60 ? "Elevated – Age ≥60 increases OA risk" : "Normal range", warn: patient.age >= 60 },
                  { label: "BMI Factor", detail: patient.bmi >= 25 ? `Overweight (${patient.bmi}) – Increased joint stress` : `Normal (${patient.bmi})`, warn: patient.bmi >= 25 },
                  { label: "Pain Severity", detail: patient.painLevel >= 7 ? "Severe – May require intervention" : patient.painLevel >= 4 ? "Moderate" : "Mild", warn: patient.painLevel >= 7 },
                  { label: "Disease Stage", detail: patient.grade !== null ? `KL Grade ${patient.grade} – ${currentKL?.label}` : "Pending", warn: (patient.grade ?? 0) >= 3 },
                ].map(item => (
                  <div key={item.label} className="flex items-start gap-2 p-2 rounded-lg border">
                    <div className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${item.warn ? "bg-warning" : "bg-success"}`} />
                    <div>
                      <p className="text-xs font-medium">{item.label}</p>
                      <p className="text-[11px] text-muted-foreground">{item.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Medical History */}
          <div className="card-clinical mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Stethoscope className="w-4 h-4 text-muted-foreground" />
              <p className="section-header">Medical History & Symptoms</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-medium mb-1.5">Medical History</p>
                <p className="text-sm text-muted-foreground leading-relaxed">{patient.history}</p>
              </div>
              <div>
                <p className="text-xs font-medium mb-1.5">Current Symptoms</p>
                <p className="text-sm text-muted-foreground leading-relaxed">{patient.symptoms}</p>
              </div>
            </div>
          </div>

          {/* AI Analysis */}
          <div className="card-clinical mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-muted-foreground" />
              <p className="section-header">AI Diagnostic Summary</p>
            </div>
            <div className="flex items-center gap-4 mb-4">
              <GradeBadge grade={displayGrade} className="!w-12 !h-12 !text-lg" />
              <div>
                <p className="text-sm font-medium">
                  {displayGrade !== null ? `Grade ${displayGrade} Osteoarthritis (${currentKL?.label})` : "Pending Analysis"}
                </p>
                {patient.aiConfidence && (
                  <p className="text-xs text-muted-foreground">
                    Confidence: {patient.aiConfidence}% · Model: DenseNet201 · Pre-processing: CLAHE + GAN Denoise
                  </p>
                )}
                <div className="flex items-center gap-2 mt-1">
                  <ConfidenceGauge value={patient.aiConfidence} />
                </div>
              </div>
            </div>

            {/* Diagnosis summary */}
            <div className="bg-primary-muted rounded-lg p-4 mt-3 space-y-3">
              <p className="text-sm leading-relaxed">
                {report?.diagnosisSummary ? (
                  report.diagnosisSummary
                ) : patient.grade !== null ? (
                  <>
                    Automated analysis indicates <strong>Grade {patient.grade} Osteoarthritis ({currentKL?.label})</strong> at{" "}
                    <strong>{patient.aiConfidence}%</strong> confidence. {currentKL?.description}
                  </>
                ) : (
                  "Analysis pending. Please upload imaging data to generate AI diagnostic summary."
                )}
              </p>
              {report?.findings && report.findings.length > 0 && (
                <ul className="space-y-1 border-t border-primary/10 pt-3">
                  {report.findings.map((f, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                      <span className="w-1 h-1 rounded-full bg-primary mt-1.5 flex-shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* KL Criteria Reference */}
            {patient.grade !== null && (
              <div className="mt-4">
                <p className="text-xs font-medium mb-2">Kellgren-Lawrence Classification Reference</p>
                <div className="space-y-1">
                  {klCriteria.map(kl => (
                    <div key={kl.grade} className={`flex items-center gap-3 px-3 py-1.5 rounded-lg text-xs transition-colors ${
                      kl.grade === patient.grade ? "bg-primary-muted border border-primary/20" : ""
                    }`}>
                      <GradeBadge grade={kl.grade} />
                      <span className="font-medium w-16">{kl.label}</span>
                      <span className="text-muted-foreground flex-1">{kl.description}</span>
                      {kl.grade === patient.grade && <span className="text-primary text-[10px] font-medium">← Current</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Key Findings Grid */}
            {patient.grade !== null && (
              <div className="mt-4 grid grid-cols-3 gap-3">
                {[
                  { label: "Joint Space", value: patient.grade >= 3 ? "Significant narrowing (>50%)" : patient.grade >= 2 ? "Moderate narrowing" : "Mild / Normal", icon: Bone },
                  { label: "Osteophytes", value: patient.grade >= 3 ? "Definite, multiple" : patient.grade >= 2 ? "Possible formation" : "Doubtful", icon: Bone },
                  { label: "Sclerosis", value: patient.grade >= 3 ? "Subchondral present" : "Not significant", icon: Shield },
                  { label: "Deformity", value: patient.grade >= 4 ? "Bone deformity present" : "No significant deformity", icon: AlertCircle },
                  { label: "Cartilage", value: patient.grade >= 2 ? "Thinning detected" : "Preserved", icon: Activity },
                  { label: "Alignment", value: patient.grade >= 4 ? "Varus/Valgus deviation" : "Within normal limits", icon: ClipboardCheck },
                ].map(finding => (
                  <div key={finding.label} className="p-3 rounded-lg border">
                    <div className="flex items-center gap-1.5 mb-1">
                      <finding.icon className="w-3 h-3 text-muted-foreground" />
                      <p className="text-xs font-medium">{finding.label}</p>
                    </div>
                    <p className="text-[11px] text-muted-foreground">{finding.value}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Diagnosis History & Progression */}
          {diagnosisHistory.length > 0 && (
            <div className="card-clinical mb-6">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-muted-foreground" />
                  <p className="section-header">Diagnosis History & Progression</p>
                </div>
                {gradeChanged && (
                  <div className={`flex items-center gap-1 text-xs font-medium ${
                    gradeTrend > 0 ? "text-destructive" : gradeTrend < 0 ? "text-success" : "text-muted-foreground"
                  }`}>
                    {gradeTrend > 0 ? <TrendingUp className="w-3.5 h-3.5" /> : gradeTrend < 0 ? <TrendingDown className="w-3.5 h-3.5" /> : <Minus className="w-3.5 h-3.5" />}
                    {gradeTrend > 0 ? "Worsening" : gradeTrend < 0 ? "Improving" : "Stable"}
                  </div>
                )}
              </div>

              {progressionData.length >= 2 && (
                <div className="mb-4">
                  <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={progressionData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={20} />
                      <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid hsl(220, 13%, 91%)" }} />
                      <Line type="monotone" dataKey="grade" stroke="hsl(217, 91%, 60%)" strokeWidth={2.5} dot={{ r: 5, fill: "hsl(217, 91%, 60%)" }} name="KL Grade" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <div className="space-y-2">
                {diagnosisHistory.map((entry, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg border">
                    <div className="text-xs text-muted-foreground w-20 flex-shrink-0">{entry.date}</div>
                    <GradeBadge grade={entry.grade} />
                    <div className="flex-1 text-xs">{entry.summary}</div>
                    <div className="text-mono text-xs text-muted-foreground">{entry.confidence}%</div>
                    {i < diagnosisHistory.length - 1 && (
                      <div className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${
                        diagnosisHistory[i + 1].grade < entry.grade ? "bg-destructive/10 text-destructive" :
                        diagnosisHistory[i + 1].grade > entry.grade ? "bg-success/10 text-success" : "bg-muted text-muted-foreground"
                      }`}>
                        {diagnosisHistory[i + 1].grade < entry.grade ? "↑ Worse" :
                         diagnosisHistory[i + 1].grade > entry.grade ? "↓ Better" : "= Same"}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {diagnosisHistory.length === 1 && (
                <div className="mt-3 p-3 rounded-lg bg-muted/50 border border-dashed">
                  <p className="text-xs text-muted-foreground text-center">Only 1 diagnosis on record. Future scans will enable progression tracking.</p>
                </div>
              )}
            </div>
          )}

          {/* Imaging */}
          <div className="card-clinical mb-6">
            <p className="section-header mb-3">Imaging & Grad-CAM Visualization</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div className="rounded-lg border overflow-hidden bg-muted/20">
                <p className="text-[10px] font-medium px-3 py-2 border-b bg-muted/40">Input scan</p>
                <div className="aspect-square max-h-64 flex items-center justify-center p-2">
                  {report?.inputImageDataUrl ? (
                    <img src={report.inputImageDataUrl} alt={report.inputFileName} className="max-h-full max-w-full object-contain" />
                  ) : (
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                      <Bone className="w-8 h-8 opacity-40" />
                      <p className="text-xs">{report?.inputFileName ?? "No image stored"}</p>
                    </div>
                  )}
                </div>
                {report?.inputFileName && (
                  <p className="text-[10px] text-muted-foreground px-3 py-2 border-t truncate">{report.inputFileName}</p>
                )}
              </div>
              <div className="rounded-lg border overflow-hidden bg-muted/20">
                <p className="text-[10px] font-medium px-3 py-2 border-b bg-muted/40">Ensemble Grad-CAM</p>
                <div className="aspect-square max-h-64 flex items-center justify-center p-2">
                  {report?.ensembleGradcamDataUrl ? (
                    <img src={report.ensembleGradcamDataUrl} alt="Ensemble Grad-CAM" className="max-h-full max-w-full object-contain" />
                  ) : (
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                      <Activity className="w-8 h-8 opacity-40" />
                      <p className="text-xs">No heatmap stored</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
            {report?.modalitySnapshots && report.modalitySnapshots.length > 1 && (
              <div className="mb-4">
                <p className="text-xs font-medium mb-2">Multi-modality analysis</p>
                <div className="space-y-3">
                  {report.modalitySnapshots.map(snap => (
                    <div key={snap.modality} className="rounded-lg border p-3 bg-muted/20">
                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        <span className="text-[10px] uppercase font-semibold tracking-wider">
                          {snap.modality === "xray" ? "X-Ray" : "MRI"}
                        </span>
                        <GradeBadge grade={snap.grade} />
                        <span className="text-[10px] text-muted-foreground">{snap.confidence.toFixed(1)}% · {snap.modelUsed}</span>
                      </div>
                      <p className="text-[10px] text-muted-foreground mb-1">{snap.inputFileName}</p>
                      <ul className="space-y-0.5">
                        {snap.findings.slice(0, 3).map((f, i) => (
                          <li key={i} className="text-[11px] text-muted-foreground">• {f}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {report?.modelResults && report.modelResults.length > 0 && (
              <div>
                <p className="text-xs font-medium mb-2">Per-model results ({report.modelResults.length})</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                  {report.modelResults.map(m => (
                    <div key={m.modelId} className="rounded-lg border overflow-hidden">
                      <div className="px-2 py-1.5 border-b bg-muted/30 flex items-center justify-between gap-1">
                        <span className="text-[9px] font-medium truncate">{m.displayName}</span>
                        <GradeBadge grade={m.grade} />
                      </div>
                      <div className="aspect-square bg-black/[0.03] p-1">
                        {m.gradcamDataUrl ? (
                          <img src={m.gradcamDataUrl} alt={m.displayName} className="w-full h-full object-contain" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-[9px] text-muted-foreground">N/A</div>
                        )}
                      </div>
                      <p className="text-[9px] text-center text-muted-foreground py-1">{m.confidence.toFixed(1)}%</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Treatment Recommendations */}
          <div className="card-clinical mb-6">
            <div className="flex items-center gap-2 mb-3">
              <ClipboardCheck className="w-4 h-4 text-muted-foreground" />
              <p className="section-header">Treatment Recommendations</p>
            </div>
            <div className="space-y-2">
              {patient.grade !== null && patient.grade >= 3 ? (
                <>
                  {[
                    "Referral to orthopedic surgery for evaluation of total knee arthroplasty (TKA)",
                    "Consider intra-articular corticosteroid or hyaluronic acid injection",
                    "Physical therapy: quadriceps strengthening and range of motion exercises",
                    `Weight management counseling (current BMI: ${patient.bmi})`,
                    "Pain management: NSAIDs, topical analgesics, or tramadol if needed",
                    "Follow-up imaging in 3 months to reassess progression",
                  ].map((rec, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 flex-shrink-0" />
                      {rec}
                    </div>
                  ))}
                </>
              ) : patient.grade !== null ? (
                <>
                  {[
                    "Conservative management: NSAIDs and lifestyle modifications",
                    "Low-impact exercise: swimming, cycling, walking",
                    "Follow-up imaging in 6 months to monitor progression",
                    "Physical therapy and joint protection techniques",
                    patient.bmi >= 25 ? `Weight loss recommended (current BMI: ${patient.bmi})` : "Maintain healthy weight and activity level",
                  ].map((rec, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 flex-shrink-0" />
                      {rec}
                    </div>
                  ))}
                </>
              ) : (
                <p className="text-sm text-muted-foreground">Awaiting diagnosis to generate recommendations.</p>
              )}
            </div>
          </div>

          {/* Complete Timeline */}
          <div className="card-clinical mb-6">
            <p className="section-header mb-3">Complete Patient Timeline</p>
            <div className="space-y-0">
              {patient.timeline.map((entry, i) => (
                <div key={i} className="flex gap-3 pb-3 relative">
                  {i < patient.timeline.length - 1 && (
                    <div className="absolute left-[7px] top-5 bottom-0 w-px bg-border" />
                  )}
                  <div className={`w-4 h-4 rounded-full flex-shrink-0 mt-0.5 border-2 ${
                    entry.type === "diagnosis" ? "border-primary bg-primary-muted" :
                    entry.type === "scan" ? "border-muted-foreground bg-muted" :
                    entry.type === "report" ? "border-success bg-success/10" :
                    "border-warning bg-warning/10"
                  }`} />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium">{entry.date}</p>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground capitalize">{entry.type}</span>
                      {entry.grade !== undefined && <GradeBadge grade={entry.grade} />}
                    </div>
                    <p className="text-sm text-muted-foreground mt-0.5">{entry.summary}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical Notes */}
          <div className="card-clinical mb-6">
            <div className="flex items-center justify-between mb-3">
              <p className="section-header">Clinical Notes</p>
              <button className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors">
                <Edit3 className="w-3 h-3" />
                Edit
              </button>
            </div>
            <textarea
              className="w-full h-32 resize-none border rounded-lg p-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/20 bg-background"
              placeholder="Add clinical notes, observations, and treatment recommendations..."
              defaultValue={patient.status === "confirmed" ? "Findings confirmed. Patient referred to Orthopedic Surgery for evaluation of total knee arthroplasty. Follow-up imaging in 6 months recommended." : ""}
            />
          </div>

          {/* Disclaimer */}
          <div className="text-center pb-8">
            <p className="text-[10px] text-muted-foreground leading-relaxed max-w-md mx-auto">
              This report was generated by KneeXpert AI diagnostic system. AI-assisted diagnoses are intended to support clinical decision-making
              and should not replace professional medical judgment. All findings should be reviewed and confirmed by a qualified radiologist or orthopedic specialist.
            </p>
          </div>
        </motion.div>
      </div>

      {/* Actions Sidebar */}
      <div className="w-52 border-l bg-muted/30 p-3 space-y-2.5 flex-shrink-0 hidden lg:block">
        <p className="section-header mb-4">Quick Actions</p>

        <button onClick={handlePreviewPdf} className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
          <Eye className="w-4 h-4" />
          Preview & Export PDF
        </button>
        <button onClick={handleDownloadPdf} className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
          <Download className="w-4 h-4" />
          Download PDF
        </button>
        <button onClick={handlePrintReport} className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
          <Printer className="w-4 h-4" />
          Print Report
        </button>
        <button className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
          <Send className="w-4 h-4" />
          Refer to Specialist
        </button>

        <div className="pt-4 border-t mt-4">
          <p className="section-header mb-3">Report Summary</p>
          <div className="space-y-2.5">
            {[
              { label: "Current Grade", value: patient.grade !== null ? `Grade ${patient.grade}` : "Pending" },
              { label: "Classification", value: currentKL?.label || "—" },
              { label: "Confidence", value: patient.aiConfidence ? `${patient.aiConfidence}%` : "—" },
              { label: "Risk Level", value: riskLevel },
              { label: "Diagnoses", value: `${diagnosisHistory.length}` },
              { label: "Progression", value: gradeChanged ? (gradeTrend > 0 ? "Worsening" : gradeTrend < 0 ? "Improving" : "Stable") : "Insufficient data" },
              { label: "Generated", value: new Date().toLocaleDateString() },
              { label: "Last Modified", value: patient.lastVisit },
            ].map(item => (
              <div key={item.label} className="text-xs">
                <p className="text-muted-foreground">{item.label}</p>
                <p className="font-medium">{item.value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="pt-4 border-t">
          <p className="section-header mb-3">Related Actions</p>
          <button
            onClick={() => navigate(`/diagnostics?patient=${patient.id}`)}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium hover:bg-muted transition-colors mb-2"
          >
            <Activity className="w-3.5 h-3.5" />
            Open in Diagnostics
          </button>
          <button
            onClick={() => navigate(`/patients`)}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium hover:bg-muted transition-colors"
          >
            <Stethoscope className="w-3.5 h-3.5" />
            View Patient Record
          </button>
        </div>
      </div>
    </div>
    </>
  );
}
