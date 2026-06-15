import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Plus, SlidersHorizontal, ChevronDown, X } from "lucide-react";
import { Patient } from "@/data/patients";
import { usePatients } from "@/context/PatientContext";
import { StatusBadge } from "@/components/StatusBadge";
import { GradeBadge } from "@/components/GradeBadge";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { PatientDetailPanel } from "@/components/PatientDetailPanel";
import { AddPatientDialog } from "@/components/AddPatientDialog";
import { cn } from "@/lib/utils";

const anim = {
  hidden: { opacity: 0, y: 6 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.25, staggerChildren: 0.04 } },
};
const item = { hidden: { opacity: 0, y: 4 }, visible: { opacity: 1, y: 0 } };

const AGE_RANGES = [
  { label: "All Ages", min: 0, max: 200 },
  { label: "<40", min: 0, max: 39 },
  { label: "40–54", min: 40, max: 54 },
  { label: "55–64", min: 55, max: 64 },
  { label: "65+", min: 65, max: 200 },
];

const BMI_RANGES = [
  { label: "All BMI", min: 0, max: 100 },
  { label: "Normal (<25)", min: 0, max: 24.9 },
  { label: "Overweight (25–30)", min: 25, max: 29.9 },
  { label: "Obese (30+)", min: 30, max: 100 },
];

const PAIN_RANGES = [
  { label: "All", min: 0, max: 10 },
  { label: "Mild (1–3)", min: 1, max: 3 },
  { label: "Moderate (4–6)", min: 4, max: 6 },
  { label: "Severe (7–10)", min: 7, max: 10 },
];

export default function PatientsPage() {
  const { patients } = usePatients();
  const [search, setSearch] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [gradeFilter, setGradeFilter] = useState<number | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [showAddPatient, setShowAddPatient] = useState(false);
  const [viewMode, setViewMode] = useState<"cards" | "table">("cards");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [ageRange, setAgeRange] = useState(0);
  const [bmiRange, setBmiRange] = useState(0);
  const [painRange, setPainRange] = useState(0);
  const [modalityFilter, setModalityFilter] = useState<string>("all");
  const [genderFilter, setGenderFilter] = useState<string>("all");
  const [confidenceFilter, setConfidenceFilter] = useState<string>("all");

  const activeFilterCount = [
    ageRange !== 0,
    bmiRange !== 0,
    painRange !== 0,
    modalityFilter !== "all",
    genderFilter !== "all",
    confidenceFilter !== "all",
  ].filter(Boolean).length;

  const clearAdvanced = () => {
    setAgeRange(0); setBmiRange(0); setPainRange(0);
    setModalityFilter("all"); setGenderFilter("all"); setConfidenceFilter("all");
  };

  const filtered = patients.filter((p) => {
    const s = search.toLowerCase();
    const matchSearch = p.name.toLowerCase().includes(s) || p.id.toLowerCase().includes(s);
    const matchGrade = gradeFilter === null || p.grade === gradeFilter;
    const matchStatus = statusFilter === "all" || p.status === statusFilter;
    const ar = AGE_RANGES[ageRange];
    const matchAge = p.age >= ar.min && p.age <= ar.max;
    const br = BMI_RANGES[bmiRange];
    const matchBmi = p.bmi >= br.min && p.bmi <= br.max;
    const pr = PAIN_RANGES[painRange];
    const matchPain = p.painLevel >= pr.min && p.painLevel <= pr.max;
    const matchModality = modalityFilter === "all" || p.modality === modalityFilter;
    const matchGender = genderFilter === "all" || p.gender === genderFilter;
    const matchConfidence = confidenceFilter === "all" ||
      (confidenceFilter === "high" && p.aiConfidence !== null && p.aiConfidence >= 90) ||
      (confidenceFilter === "medium" && p.aiConfidence !== null && p.aiConfidence >= 70 && p.aiConfidence < 90) ||
      (confidenceFilter === "low" && (p.aiConfidence === null || p.aiConfidence < 70));
    return matchSearch && matchGrade && matchStatus && matchAge && matchBmi && matchPain && matchModality && matchGender && matchConfidence;
  });

  return (
    <div className="flex min-h-screen">
      <div className="flex-1 overflow-auto">
        <motion.div variants={anim} initial="hidden" animate="visible" className="p-3 sm:p-4 lg:p-5 space-y-4 max-w-[1400px] mx-auto">
          {/* Header */}
          <motion.div variants={item} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Patients</h1>
              <p className="text-xs text-muted-foreground mt-0.5">Electronic Health Records & Case Management</p>
            </div>
            <button
              onClick={() => setShowAddPatient(true)}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors self-start"
            >
              <Plus className="w-4 h-4" />New Patient
            </button>
          </motion.div>

          {/* Filters */}
          <motion.div variants={item} className="space-y-3">
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  placeholder="Search by name or ID..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 transition"
                />
              </div>
               <div className="flex items-center gap-2 flex-wrap">
                {/* Status filter */}
                <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
                  {["all", "pending", "analyzed", "confirmed", "flagged"].map((s) => (
                    <button
                      key={s}
                      onClick={() => setStatusFilter(s)}
                      className={cn(
                        "px-2.5 py-1.5 rounded-md text-xs font-medium capitalize transition-all",
                        statusFilter === s ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                {/* Grade filter */}
                <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
                  <button onClick={() => setGradeFilter(null)} className={cn("px-2 py-1.5 rounded-md text-xs font-medium transition-all", gradeFilter === null ? "bg-background text-foreground shadow-sm" : "text-muted-foreground")}>All</button>
                  {[0, 1, 2, 3, 4].map((g) => (
                    <button key={g} onClick={() => setGradeFilter(g)} className={cn("px-2 py-1.5 rounded-md text-xs font-medium transition-all", gradeFilter === g ? "bg-background text-foreground shadow-sm" : "text-muted-foreground")}>G{g}</button>
                  ))}
                </div>
                {/* Advanced toggle */}
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className={cn(
                    "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border",
                    showAdvanced || activeFilterCount > 0
                      ? "bg-primary/10 text-primary border-primary/20"
                      : "bg-muted text-muted-foreground border-transparent hover:text-foreground"
                  )}
                >
                  <SlidersHorizontal className="w-3.5 h-3.5" />
                  Filters
                  {activeFilterCount > 0 && (
                    <span className="ml-0.5 w-4 h-4 rounded-full bg-primary text-primary-foreground text-[10px] flex items-center justify-center">{activeFilterCount}</span>
                  )}
                  <ChevronDown className={cn("w-3 h-3 transition-transform", showAdvanced && "rotate-180")} />
                </button>
              </div>
            </div>

            {/* Advanced Filters Panel */}
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="rounded-xl border bg-muted/30 p-3 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-foreground">Clinical Filters</span>
                  {activeFilterCount > 0 && (
                    <button onClick={clearAdvanced} className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                      <X className="w-3 h-3" /> Clear all
                    </button>
                  )}
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                  {/* Age Range */}
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">Age Range</label>
                    <select value={ageRange} onChange={(e) => setAgeRange(Number(e.target.value))} className="w-full rounded-lg border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring/20">
                      {AGE_RANGES.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
                    </select>
                  </div>
                  {/* BMI */}
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">BMI Category</label>
                    <select value={bmiRange} onChange={(e) => setBmiRange(Number(e.target.value))} className="w-full rounded-lg border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring/20">
                      {BMI_RANGES.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
                    </select>
                  </div>
                  {/* Pain Level */}
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">Pain Level</label>
                    <select value={painRange} onChange={(e) => setPainRange(Number(e.target.value))} className="w-full rounded-lg border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring/20">
                      {PAIN_RANGES.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
                    </select>
                  </div>
                  {/* Modality */}
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">Modality</label>
                    <select value={modalityFilter} onChange={(e) => setModalityFilter(e.target.value)} className="w-full rounded-lg border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring/20">
                      <option value="all">All</option>
                      <option value="xray">X-Ray</option>
                      <option value="mri">MRI</option>
                    </select>
                  </div>
                  {/* Gender */}
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">Gender</label>
                    <select value={genderFilter} onChange={(e) => setGenderFilter(e.target.value)} className="w-full rounded-lg border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring/20">
                      <option value="all">All</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                    </select>
                  </div>
                  {/* AI Confidence */}
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">AI Confidence</label>
                    <select value={confidenceFilter} onChange={(e) => setConfidenceFilter(e.target.value)} className="w-full rounded-lg border bg-background px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring/20">
                      <option value="all">All</option>
                      <option value="high">High (≥90%)</option>
                      <option value="medium">Medium (70–89%)</option>
                      <option value="low">Low (&lt;70%)</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}

            <p className="text-xs text-muted-foreground">{filtered.length} patient{filtered.length !== 1 ? "s" : ""}</p>
          </motion.div>

          {/* Patient Cards */}
          <motion.div variants={item} className="space-y-2">
            {filtered.map((patient) => (
              <div
                key={patient.id}
                onClick={() => setSelectedPatient(patient)}
                className={cn(
                  "card-clinical cursor-pointer transition-all hover:shadow-card-hover",
                  selectedPatient?.id === patient.id && "ring-2 ring-primary/20"
                )}
              >
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-sm font-medium text-primary flex-shrink-0">
                      {patient.name.split(" ").slice(-1)[0][0]}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="text-sm font-medium">{patient.name}</p>
                        <span className="text-mono text-[10px] text-muted-foreground">{patient.id}</span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {patient.age}yo · {patient.gender} · BMI {patient.bmi} · Pain {patient.painLevel}/10
                      </p>
                    </div>
                  </div>
                  <div className="hidden sm:flex items-center gap-3 flex-shrink-0">
                    <GradeBadge grade={patient.grade} />
                    <ConfidenceGauge value={patient.aiConfidence} />
                    <StatusBadge status={patient.status} />
                    <span className="text-xs text-muted-foreground whitespace-nowrap">{patient.lastVisit}</span>
                  </div>
                </div>
                {/* Mobile-only badges */}
                <div className="flex sm:hidden items-center gap-2 mt-2 flex-wrap">
                  <GradeBadge grade={patient.grade} />
                  <StatusBadge status={patient.status} />
                  <span className="text-xs text-muted-foreground">{patient.lastVisit}</span>
                </div>
              </div>
            ))}
            {filtered.length === 0 && (
              <div className="py-16 text-center text-sm text-muted-foreground">
                No patients match your criteria
              </div>
            )}
          </motion.div>
        </motion.div>
      </div>

      {selectedPatient && <PatientDetailPanel patient={selectedPatient} onClose={() => setSelectedPatient(null)} />}
      <AddPatientDialog open={showAddPatient} onClose={() => setShowAddPatient(false)} />
    </div>
  );
}
