import { useState } from "react";
import { motion } from "framer-motion";
import { Search, FileText, Clock, AlertTriangle, CheckCircle2, Eye, Download, TrendingUp, TrendingDown, Minus, ArrowRight } from "lucide-react";
import { Patient } from "@/data/patients";
import { usePatients } from "@/context/PatientContext";
import { GradeBadge } from "@/components/GradeBadge";
import { StatusBadge } from "@/components/StatusBadge";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { useNavigate } from "react-router-dom";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { cn } from "@/lib/utils";

const anim = {
  hidden: { opacity: 0, y: 6 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.25, staggerChildren: 0.04 } },
};
const item = { hidden: { opacity: 0, y: 4 }, visible: { opacity: 1, y: 0 } };

const recentlyViewed = ["PT-8842", "PT-6105", "PT-5530"];

function getProgression(patient: Patient) {
  const diagnoses = patient.timeline.filter(e => e.type === "diagnosis" && e.grade !== undefined);
  if (diagnoses.length < 2) return { label: "Insufficient data", trend: 0 };
  const first = diagnoses[diagnoses.length - 1].grade!;
  const last = diagnoses[0].grade!;
  const diff = last - first;
  if (diff > 0) return { label: "Worsening", trend: 1 };
  if (diff < 0) return { label: "Improving", trend: -1 };
  return { label: "Stable", trend: 0 };
}

export default function ReportsHubPage() {
  const navigate = useNavigate();
  const { patients } = usePatients();
  const [search, setSearch] = useState("");
  const [tabFilter, setTabFilter] = useState<"all" | "review" | "pending" | "completed" | "recent">("all");

  const needsReview = patients.filter(p => p.status === "flagged" || (p.aiConfidence !== null && p.aiConfidence < 80));
  const pendingReports = patients.filter(p => p.status === "pending" || !p.report?.doctorConfirmed);
  const completedReports = patients.filter(p => p.report?.doctorConfirmed);

  const statusBreakdown = [
    { name: "Confirmed", value: completedReports.length, color: "hsl(160, 84%, 39%)" },
    { name: "Analyzed", value: patients.filter(p => p.status === "analyzed").length, color: "hsl(217, 91%, 60%)" },
    { name: "Flagged", value: patients.filter(p => p.status === "flagged").length, color: "hsl(38, 92%, 50%)" },
    { name: "Pending", value: pendingReports.length, color: "hsl(215, 16%, 70%)" },
  ];

  const getFiltered = () => {
    let list = patients;
    if (tabFilter === "review") list = needsReview;
    else if (tabFilter === "pending") list = pendingReports;
    else if (tabFilter === "completed") list = completedReports;
    else if (tabFilter === "recent") list = patients.filter(p => recentlyViewed.includes(p.id));
    if (search) list = list.filter(p => p.name.toLowerCase().includes(search.toLowerCase()) || p.id.toLowerCase().includes(search.toLowerCase()));
    return list;
  };
  const filtered = getFiltered();

  const tabs = [
    { id: "all" as const, label: "All", count: patients.length },
    { id: "recent" as const, label: "Recent", count: recentlyViewed.length },
    { id: "review" as const, label: "Review", count: needsReview.length },
    { id: "pending" as const, label: "Pending", count: pendingReports.length },
    { id: "completed" as const, label: "Completed", count: completedReports.length },
  ];

  return (
    <div className="min-h-screen overflow-auto">
      <motion.div variants={anim} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5 max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div variants={item} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Reports</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Clinical diagnostic reports & documentation</p>
          </div>
          <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors self-start">
            <Download className="w-4 h-4" />Batch Export
          </button>
        </motion.div>

        {/* Summary Cards */}
        <motion.div variants={item} className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { icon: FileText, label: "Total Reports", value: patients.length, color: "bg-primary/10", iconColor: "text-primary" },
            { icon: AlertTriangle, label: "Needs Review", value: needsReview.length, color: "bg-warning/10", iconColor: "text-warning" },
            { icon: Clock, label: "Pending", value: pendingReports.length, color: "bg-muted", iconColor: "text-muted-foreground" },
            { icon: CheckCircle2, label: "Completed", value: completedReports.length, color: "bg-success/10", iconColor: "text-success" },
          ].map(c => (
            <div key={c.label} className="card-clinical">
              <div className="flex items-center gap-3">
                <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center", c.color)}>
                  <c.icon className={cn("w-4 h-4", c.iconColor)} />
                </div>
                <div>
                  <p className="text-xl font-semibold">{c.value}</p>
                  <p className="text-[10px] text-muted-foreground">{c.label}</p>
                </div>
              </div>
            </div>
          ))}
        </motion.div>

        {/* Chart + Priority Queue */}
        <motion.div variants={item} className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="card-clinical">
            <p className="section-header mb-3">Status Breakdown</p>
            <ResponsiveContainer width="100%" height={150}>
              <PieChart>
                <Pie data={statusBreakdown} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={60} strokeWidth={2} stroke="hsl(0, 0%, 100%)">
                  {statusBreakdown.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-2 mt-2 justify-center">
              {statusBreakdown.map(s => (
                <div key={s.name} className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full" style={{ background: s.color }} />
                  <span className="text-[10px] text-muted-foreground">{s.name} ({s.value})</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card-clinical lg:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <p className="section-header">Priority Review Queue</p>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-warning/10 text-warning font-medium">{needsReview.length} cases</span>
            </div>
            {needsReview.length > 0 ? (
              <div className="space-y-2">
                {needsReview.map(p => (
                  <div key={p.id} onClick={() => navigate(`/reports/${p.id}`)} className="flex items-center gap-3 p-3 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors">
                    <div className="w-8 h-8 rounded-full bg-warning/10 flex items-center justify-center flex-shrink-0">
                      <AlertTriangle className="w-3.5 h-3.5 text-warning" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{p.name} <span className="text-mono text-[10px] text-muted-foreground">{p.id}</span></p>
                      <p className="text-xs text-muted-foreground mt-0.5">{p.status === "flagged" ? "Flagged for review" : `Confidence: ${p.aiConfidence}%`}</p>
                    </div>
                    <GradeBadge grade={p.grade} />
                    <ArrowRight className="w-4 h-4 text-muted-foreground" />
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-6">All reviewed ✓</p>
            )}
          </div>
        </motion.div>

        {/* Reports Table */}
        <motion.div variants={item} className="card-clinical !p-0 overflow-hidden">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 px-4 pt-4 pb-3 border-b">
            <div className="flex items-center gap-1 overflow-x-auto">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setTabFilter(tab.id)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap",
                    tabFilter === tab.id ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}
                >
                  {tab.label} <span className="ml-1 text-[10px] opacity-70">{tab.count}</span>
                </button>
              ))}
            </div>
            <div className="sm:ml-auto relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
              <input
                placeholder="Search reports..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="pl-9 pr-3 py-1.5 rounded-lg border bg-background text-xs w-full sm:w-48 focus:outline-none focus:ring-2 focus:ring-ring/20 transition"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px]">
              <thead>
                <tr className="border-b">
                  <th className="text-left px-4 py-3 section-header">Patient</th>
                  <th className="text-left px-4 py-3 section-header">Grade</th>
                  <th className="text-left px-4 py-3 section-header">Confidence</th>
                  <th className="text-left px-4 py-3 section-header">Progression</th>
                  <th className="text-left px-4 py-3 section-header">Status</th>
                  <th className="text-left px-4 py-3 section-header">Date</th>
                  <th className="text-left px-4 py-3 section-header"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(patient => {
                  const report = patient.report;
                  const prog = getProgression(patient);
                  const isRecent = recentlyViewed.includes(patient.id);
                  return (
                    <tr key={patient.id} onClick={() => navigate(`/reports/${patient.id}`)} className="border-b last:border-0 cursor-pointer hover:bg-muted/50 transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium text-primary flex-shrink-0">
                            {patient.name.split(" ").slice(-1)[0][0]}
                          </div>
                          <div>
                            <p className="text-sm font-medium">{patient.name}</p>
                            <p className="text-mono text-[10px] text-muted-foreground">{patient.id}</p>
                          </div>
                          {isRecent && <Eye className="w-3 h-3 text-primary ml-1" />}
                        </div>
                      </td>
                      <td className="px-4 py-3"><GradeBadge grade={report?.finalGrade ?? patient.grade} /></td>
                      <td className="px-4 py-3"><ConfidenceGauge value={report?.aiConfidence ?? patient.aiConfidence} /></td>
                      <td className="px-4 py-3">
                        <span className={cn("flex items-center gap-1 text-xs font-medium", prog.trend > 0 ? "text-destructive" : prog.trend < 0 ? "text-success" : "text-muted-foreground")}>
                          {prog.trend > 0 ? <TrendingUp className="w-3 h-3" /> : prog.trend < 0 ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                          {prog.label}
                        </span>
                      </td>
                      <td className="px-4 py-3"><StatusBadge status={patient.status} /></td>
                      <td className="px-4 py-3 text-xs text-muted-foreground">
                        {report?.updatedAt || "—"}
                        {report?.version ? <span className="block text-[10px] text-muted-foreground/70">v{report.version}</span> : null}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1">
                          {report?.doctorConfirmed && <span className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">Confirmed</span>}
                          <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        </div>
                      </td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && (
                  <tr><td colSpan={7} className="px-4 py-12 text-center text-sm text-muted-foreground">No reports found</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
