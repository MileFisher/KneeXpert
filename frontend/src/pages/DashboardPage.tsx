import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Plus, TrendingUp, Users, Clock, AlertTriangle, ArrowUpRight, ArrowDownRight, Scan, FileText } from "lucide-react";
import { Patient } from "@/data/patients";
import { usePatients } from "@/context/PatientContext";
import { StatusBadge } from "@/components/StatusBadge";
import { GradeBadge } from "@/components/GradeBadge";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { PatientDetailPanel } from "@/components/PatientDetailPanel";
import { AddPatientDialog } from "@/components/AddPatientDialog";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, AreaChart, Area, CartesianGrid, LineChart, Line } from "recharts";
import { useNavigate } from "react-router-dom";

const anim = {
  hidden: { opacity: 0, y: 6 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.25, staggerChildren: 0.04 } },
};
const item = { hidden: { opacity: 0, y: 4 }, visible: { opacity: 1, y: 0 } };

const gradeDistribution = [
  { grade: "G0", count: 12, color: "hsl(215, 16%, 80%)" },
  { grade: "G1", count: 28, color: "hsl(217, 91%, 80%)" },
  { grade: "G2", count: 35, color: "hsl(217, 91%, 65%)" },
  { grade: "G3", count: 32, color: "hsl(217, 91%, 50%)" },
  { grade: "G4", count: 20, color: "hsl(217, 91%, 35%)" },
];

const weeklyTrend = [
  { day: "Mon", scans: 12, confirmed: 9 },
  { day: "Tue", scans: 18, confirmed: 14 },
  { day: "Wed", scans: 15, confirmed: 12 },
  { day: "Thu", scans: 22, confirmed: 18 },
  { day: "Fri", scans: 19, confirmed: 15 },
  { day: "Sat", scans: 8, confirmed: 6 },
  { day: "Sun", scans: 5, confirmed: 4 },
];

const confidenceTrend = [
  { month: "Oct", avg: 87.2 },
  { month: "Nov", avg: 88.5 },
  { month: "Dec", avg: 89.1 },
  { month: "Jan", avg: 90.3 },
  { month: "Feb", avg: 91.0 },
  { month: "Mar", avg: 91.4 },
];

const stats = [
  { label: "Total Patients", value: "127", icon: Users, trend: "+12", up: true },
  { label: "Pending", value: "8", icon: Clock, trend: "3 urgent", up: false },
  { label: "Avg Confidence", value: "91.4%", icon: TrendingUp, trend: "+2.1%", up: true },
  { label: "Flagged", value: "4", icon: AlertTriangle, trend: "2 new", up: false },
  { label: "Scans/Week", value: "99", icon: Scan, trend: "+15%", up: true },
  { label: "Reports", value: "43", icon: FileText, trend: "+8", up: true },
];

export default function DashboardPage() {
  const { patients } = usePatients();
  const [search, setSearch] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [showAddPatient, setShowAddPatient] = useState(false);
  const navigate = useNavigate();

  const filtered = patients.filter((p) => {
    const s = search.toLowerCase();
    const matchSearch = p.name.toLowerCase().includes(s) || p.id.toLowerCase().includes(s);
    const matchStatus = statusFilter === "all" || p.status === statusFilter;
    return matchSearch && matchStatus;
  });

  return (
    <div className="flex min-h-screen">
      <div className="flex-1 overflow-auto">
        <motion.div variants={anim} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5 max-w-[1400px] mx-auto">
          {/* Header */}
          <motion.div variants={item} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Dashboard</h1>
              <p className="text-xs text-muted-foreground mt-0.5">
                {new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" })}
              </p>
            </div>
            <button
              onClick={() => setShowAddPatient(true)}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors self-start"
            >
              <Plus className="w-4 h-4" />New Patient
            </button>
          </motion.div>

          {/* Stats */}
          <motion.div variants={item} className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {stats.map((s) => (
              <div key={s.label} className="card-clinical">
                <div className="flex items-center justify-between mb-2">
                  <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                    <s.icon className="w-4 h-4 text-primary" />
                  </div>
                  <span className={`text-[10px] font-medium flex items-center gap-0.5 ${s.up ? "text-success" : "text-warning"}`}>
                    {s.up ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                    {s.trend}
                  </span>
                </div>
                <p className="text-lg font-semibold">{s.value}</p>
                <p className="text-[10px] text-muted-foreground">{s.label}</p>
              </div>
            ))}
          </motion.div>

          {/* Charts */}
          <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="card-clinical">
              <p className="section-header mb-3">OA Grade Distribution</p>
              <ResponsiveContainer width="100%" height={170}>
                <BarChart data={gradeDistribution} barCategoryGap="20%">
                  <XAxis dataKey="grade" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={25} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid hsl(220, 13%, 91%)" }} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {gradeDistribution.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card-clinical">
              <p className="section-header mb-3">Weekly Scans</p>
              <ResponsiveContainer width="100%" height={170}>
                <AreaChart data={weeklyTrend}>
                  <defs>
                    <linearGradient id="scanG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" />
                  <XAxis dataKey="day" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={25} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                  <Area type="monotone" dataKey="scans" stroke="hsl(217, 91%, 60%)" fill="url(#scanG)" strokeWidth={2} />
                  <Area type="monotone" dataKey="confirmed" stroke="hsl(160, 84%, 39%)" fill="transparent" strokeWidth={2} strokeDasharray="4 4" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="card-clinical">
              <p className="section-header mb-3">AI Confidence Trend</p>
              <ResponsiveContainer width="100%" height={170}>
                <LineChart data={confidenceTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" />
                  <XAxis dataKey="month" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis domain={[85, 95]} tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: number) => [`${v}%`, "Confidence"]} />
                  <Line type="monotone" dataKey="avg" stroke="hsl(217, 91%, 60%)" strokeWidth={2.5} dot={{ r: 3, fill: "hsl(217, 91%, 60%)" }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Patient Table — full width without activity sidebar */}
          <motion.div variants={item}>
            <div className="card-clinical !p-0 overflow-hidden">
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 p-4 border-b">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    placeholder="Search patients..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 transition"
                  />
                </div>
                <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5 overflow-x-auto">
                  {["all", "pending", "analyzed", "confirmed", "flagged"].map((s) => (
                    <button
                      key={s}
                      onClick={() => setStatusFilter(s)}
                      className={`px-2.5 py-1.5 rounded-md text-xs font-medium capitalize transition-all whitespace-nowrap ${
                        statusFilter === s ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px]">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left px-4 py-3 section-header">Patient</th>
                      <th className="text-left px-4 py-3 section-header">Age</th>
                      <th className="text-left px-4 py-3 section-header">Grade</th>
                      <th className="text-left px-4 py-3 section-header">Confidence</th>
                      <th className="text-left px-4 py-3 section-header">Status</th>
                      <th className="text-left px-4 py-3 section-header">Last Visit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((patient) => (
                      <tr
                        key={patient.id}
                        onClick={() => setSelectedPatient(patient)}
                        className={`border-b last:border-0 cursor-pointer transition-colors hover:bg-muted/50 ${
                          selectedPatient?.id === patient.id ? "bg-primary/5" : ""
                        }`}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{patient.name}</span>
                            <span className="text-mono text-[10px] text-muted-foreground">{patient.id}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">{patient.age}</td>
                        <td className="px-4 py-3"><GradeBadge grade={patient.grade} /></td>
                        <td className="px-4 py-3"><ConfidenceGauge value={patient.aiConfidence} /></td>
                        <td className="px-4 py-3"><StatusBadge status={patient.status} /></td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">{patient.lastVisit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>

      {selectedPatient && <PatientDetailPanel patient={selectedPatient} onClose={() => setSelectedPatient(null)} />}
      <AddPatientDialog open={showAddPatient} onClose={() => setShowAddPatient(false)} />
    </div>
  );
}
