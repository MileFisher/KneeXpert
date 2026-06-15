import { useState } from "react";
import { motion } from "framer-motion";
import {
  Users, Eye, Activity, Globe, TrendingUp, ArrowUpRight, ArrowDownRight,
  Monitor, Clock, BarChart3, Shield, Database, HardDrive, Search,
  CheckCircle2, AlertTriangle, Server, Zap, Brain, Cpu, Layers,
  Upload, Download, Image, FileImage, Check, RefreshCw,
  ChevronRight, ToggleLeft, ToggleRight, Plus, Filter,
  UserPlus, UserX, MoreHorizontal, Mail
} from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, PieChart, Pie, Cell } from "recharts";
import { cn } from "@/lib/utils";

const containerVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] as const, staggerChildren: 0.05 } },
};
const itemVariants = {
  hidden: { opacity: 0, y: 4 },
  visible: { opacity: 1, y: 0 },
};

// --- Traffic Data ---
const trafficData = [
  { date: "Mar 1", visitors: 42, pageViews: 128 },
  { date: "Mar 5", visitors: 58, pageViews: 187 },
  { date: "Mar 10", visitors: 71, pageViews: 234 },
  { date: "Mar 15", visitors: 65, pageViews: 198 },
  { date: "Mar 20", visitors: 89, pageViews: 312 },
  { date: "Mar 25", visitors: 97, pageViews: 345 },
  { date: "Mar 26", visitors: 103, pageViews: 367 },
];

const pageViewsData = [
  { page: "Dashboard", views: 1247, pct: 31 },
  { page: "Diagnostics", views: 986, pct: 25 },
  { page: "Patients", views: 743, pct: 19 },
  { page: "Reports", views: 521, pct: 13 },
  { page: "Settings", views: 312, pct: 8 },
  { page: "Admin", views: 156, pct: 4 },
];

const deviceBreakdown = [
  { name: "Desktop", value: 68, color: "hsl(217, 91%, 60%)" },
  { name: "Tablet", value: 22, color: "hsl(217, 91%, 78%)" },
  { name: "Mobile", value: 10, color: "hsl(215, 16%, 75%)" },
];

const activeHours = [
  { hour: "6AM", active: 2 }, { hour: "8AM", active: 8 }, { hour: "10AM", active: 14 },
  { hour: "12PM", active: 11 }, { hour: "2PM", active: 16 }, { hour: "4PM", active: 12 },
  { hour: "6PM", active: 7 }, { hour: "8PM", active: 4 }, { hour: "10PM", active: 1 },
];

// --- User Data ---
const userAccounts = [
  { name: "Dr. Quốc Châu", role: "Admin", email: "chau.nguyen@hmu.edu.vn", lastActive: "Just now", status: "online", scans: 142, joinDate: "2025-06-15" },
  { name: "Dr. Minh Tuấn", role: "Radiologist", email: "tuan.nguyen@hmu.edu.vn", lastActive: "15 min ago", status: "online", scans: 89, joinDate: "2025-08-20" },
  { name: "Dr. Thanh Hà", role: "Orthopedist", email: "ha.le@hmu.edu.vn", lastActive: "2 hrs ago", status: "offline", scans: 56, joinDate: "2025-09-10" },
  { name: "Dr. Phương Anh", role: "Resident", email: "anh.pham@hmu.edu.vn", lastActive: "1 day ago", status: "offline", scans: 23, joinDate: "2025-11-01" },
  { name: "Nurse Linh", role: "Staff", email: "linh.tran@hmu.edu.vn", lastActive: "5 hrs ago", status: "offline", scans: 0, joinDate: "2026-01-15" },
  { name: "Dr. Hoàng Long", role: "Radiologist", email: "long.hoang@hmu.edu.vn", lastActive: "3 days ago", status: "offline", scans: 34, joinDate: "2025-10-05" },
  { name: "Dr. Kim Ngân", role: "Resident", email: "ngan.kim@hmu.edu.vn", lastActive: "6 hrs ago", status: "offline", scans: 12, joinDate: "2026-02-01" },
];

// --- System Logs ---
const systemLogs = [
  { time: "10:42 AM", event: "Model v3.2.1 inference completed", type: "success" as const, detail: "Patient PT-8842" },
  { time: "10:38 AM", event: "New user login: Dr. Minh Tuấn", type: "info" as const, detail: "IP: 192.168.1.45" },
  { time: "10:15 AM", event: "Database backup completed", type: "success" as const, detail: "Size: 2.4 GB" },
  { time: "09:50 AM", event: "High memory usage detected", type: "warning" as const, detail: "87% utilization" },
  { time: "09:30 AM", event: "Edge function deployed", type: "info" as const, detail: "report-generator v2" },
  { time: "08:15 AM", event: "Model retraining started", type: "info" as const, detail: "Epoch 1/150" },
  { time: "07:00 AM", event: "Daily cron: data validation", type: "success" as const, detail: "0 issues found" },
  { time: "06:00 AM", event: "SSL certificate renewed", type: "success" as const, detail: "Valid until 2027-03-26" },
];

const kpiStats = [
  { label: "Total Visitors", value: "1,247", icon: Eye, trend: "+18%", up: true },
  { label: "Active Accounts", value: "12", icon: Users, trend: "+3", up: true },
  { label: "Page Views", value: "4,523", icon: Globe, trend: "+24%", up: true },
  { label: "Avg. Session", value: "8m 42s", icon: Clock, trend: "+1m", up: true },
  { label: "Uptime", value: "99.97%", icon: Server, trend: "0.02%", up: false },
  { label: "API Calls", value: "12.4K", icon: Zap, trend: "+32%", up: true },
];

// --- AI Training Data ---
const trainingStats = {
  totalImages: 2847, xrayImages: 2103, mriImages: 744,
  labeledImages: 2561, unlabeledImages: 286, doctorOverrides: 43,
  lastTrainingDate: "2026-03-10", currentEpoch: 150, modelVersion: "v3.2.1",
};
const datasetBreakdown = [
  { grade: 0, xray: 312, mri: 98, total: 410, label: "Normal" },
  { grade: 1, xray: 487, mri: 156, total: 643, label: "Doubtful" },
  { grade: 2, xray: 582, mri: 201, total: 783, label: "Minimal" },
  { grade: 3, xray: 498, mri: 178, total: 676, label: "Moderate" },
  { grade: 4, xray: 224, mri: 111, total: 335, label: "Severe" },
];
const recentOverrides = [
  { id: "PT-5530", date: "2026-03-10", aiGrade: 1, doctorGrade: 2, modality: "X-ray", status: "pending", doctor: "Dr. Quốc Châu" },
  { id: "PT-8842", date: "2026-03-08", aiGrade: 2, doctorGrade: 3, modality: "X-ray", status: "approved", doctor: "Dr. Minh Tuấn" },
  { id: "PT-7291", date: "2026-03-05", aiGrade: 3, doctorGrade: 3, modality: "MRI", status: "approved", doctor: "Dr. Quốc Châu" },
  { id: "PT-6105", date: "2026-03-01", aiGrade: 3, doctorGrade: 4, modality: "X-ray", status: "approved", doctor: "Dr. Thanh Hà" },
];
const modelVersions = [
  { version: "v3.2.1", date: "2026-03-10", accuracy: "94.2%", dataset: "2561 images", status: "active" },
  { version: "v3.1.0", date: "2026-02-15", accuracy: "93.1%", dataset: "2340 images", status: "archived" },
  { version: "v3.0.0", date: "2026-01-20", accuracy: "91.8%", dataset: "2105 images", status: "archived" },
  { version: "v2.5.0", date: "2025-12-01", accuracy: "89.5%", dataset: "1820 images", status: "archived" },
];

const Toggle = ({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) => (
  <button onClick={() => onChange(!value)} className="flex-shrink-0">
    {value ? <ToggleRight className="w-8 h-8 text-primary" /> : <ToggleLeft className="w-8 h-8 text-muted-foreground" />}
  </button>
);

// ============================================================
// Overview Page
// ============================================================
export function AdminOverviewPage() {
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">System analytics and performance metrics</p>
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {kpiStats.map(stat => (
          <div key={stat.label} className="card-clinical">
            <div className="flex items-center justify-between mb-2">
              <div className="w-8 h-8 rounded-lg bg-primary-muted flex items-center justify-center">
                <stat.icon className="w-4 h-4 text-primary" />
              </div>
              <div className={cn("flex items-center gap-0.5 text-[10px] sm:text-xs font-medium", stat.up ? "text-success" : "text-warning")}>
                {stat.up ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                {stat.trend}
              </div>
            </div>
            <p className="text-lg sm:text-xl font-semibold">{stat.value}</p>
            <p className="text-[10px] sm:text-[11px] text-muted-foreground">{stat.label}</p>
          </div>
        ))}
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card-clinical lg:col-span-2">
          <p className="section-header mb-3">Traffic Overview</p>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={trafficData}>
              <defs>
                <linearGradient id="adminGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid hsl(220, 13%, 91%)" }} />
              <Area type="monotone" dataKey="pageViews" stroke="hsl(217, 91%, 60%)" fill="url(#adminGrad)" strokeWidth={2} name="Page Views" />
              <Area type="monotone" dataKey="visitors" stroke="hsl(160, 84%, 39%)" fill="transparent" strokeWidth={2} strokeDasharray="4 4" name="Visitors" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="card-clinical">
          <p className="section-header mb-3">Device Breakdown</p>
          <ResponsiveContainer width="100%" height={140}>
            <PieChart>
              <Pie data={deviceBreakdown} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={55} strokeWidth={2} stroke="hsl(0,0%,100%)">
                {deviceBreakdown.map((entry, i) => <Cell key={i} fill={entry.color} />)}
              </Pie>
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid hsl(220, 13%, 91%)" }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-col gap-1.5 mt-2">
            {deviceBreakdown.map(d => (
              <div key={d.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                  <span className="text-xs text-muted-foreground">{d.name}</span>
                </div>
                <span className="text-xs font-medium">{d.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card-clinical">
          <p className="section-header mb-3">Page Views</p>
          <div className="space-y-2.5">
            {pageViewsData.map(p => (
              <div key={p.page} className="flex items-center gap-3">
                <span className="text-xs w-24 text-muted-foreground">{p.page}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full" style={{ width: `${p.pct}%` }} />
                </div>
                <span className="text-xs font-medium w-12 text-right">{p.views.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="card-clinical">
          <p className="section-header mb-3">Peak Activity Hours</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={activeHours} barCategoryGap="25%">
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={20} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid hsl(220, 13%, 91%)" }} />
              <Bar dataKey="active" radius={[4, 4, 0, 0]} fill="hsl(217, 91%, 60%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// Users Page
// ============================================================
export function AdminUsersPage() {
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");

  const roles = ["all", "Admin", "Radiologist", "Orthopedist", "Resident", "Staff"];
  const filtered = userAccounts.filter(u => {
    const matchSearch = u.name.toLowerCase().includes(search.toLowerCase()) || u.email.toLowerCase().includes(search.toLowerCase());
    const matchRole = roleFilter === "all" || u.role === roleFilter;
    return matchSearch && matchRole;
  });

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">User Management</h1>
          <p className="text-xs text-muted-foreground mt-0.5">{userAccounts.length} registered accounts · {userAccounts.filter(u => u.status === "online").length} online now</p>
        </div>
        <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors self-start">
          <UserPlus className="w-4 h-4" />
          Add User
        </button>
      </motion.div>

      {/* Stats */}
      <motion.div variants={itemVariants} className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Total Users", value: userAccounts.length, icon: Users },
          { label: "Online Now", value: userAccounts.filter(u => u.status === "online").length, icon: Activity },
          { label: "Doctors", value: userAccounts.filter(u => ["Radiologist", "Orthopedist", "Resident"].includes(u.role)).length, icon: Shield },
          { label: "Total Scans", value: userAccounts.reduce((s, u) => s + u.scans, 0), icon: BarChart3 },
        ].map(s => (
          <div key={s.label} className="card-clinical">
            <div className="flex items-center gap-2 mb-1">
              <s.icon className="w-4 h-4 text-primary" />
              <span className="text-[11px] text-muted-foreground">{s.label}</span>
            </div>
            <p className="text-xl font-semibold">{s.value}</p>
          </div>
        ))}
      </motion.div>

      {/* Search + Filter */}
      <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search users by name or email..."
            className="w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/30"
          />
        </div>
        <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
          {roles.map(r => (
            <button
              key={r}
              onClick={() => setRoleFilter(r)}
              className={cn("px-2.5 py-1.5 rounded-md text-xs font-medium transition-all", roleFilter === r ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground")}
            >
              {r === "all" ? "All" : r}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Users Table */}
      <motion.div variants={itemVariants} className="card-clinical !p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left px-4 py-3 section-header">User</th>
                <th className="text-left px-4 py-3 section-header">Role</th>
                <th className="text-left px-4 py-3 section-header">Email</th>
                <th className="text-left px-4 py-3 section-header">Scans</th>
                <th className="text-left px-4 py-3 section-header">Joined</th>
                <th className="text-left px-4 py-3 section-header">Last Active</th>
                <th className="text-left px-4 py-3 section-header">Status</th>
                <th className="text-left px-4 py-3 section-header"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(user => (
                <tr key={user.email} className="border-b last:border-0 hover:bg-muted/30 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-primary-muted flex items-center justify-center text-xs font-medium text-primary">
                        {user.name.charAt(0)}
                      </div>
                      <span className="text-sm font-medium">{user.name}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={cn("text-[10px] px-2 py-0.5 rounded-full font-medium",
                      user.role === "Admin" ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
                    )}>{user.role}</span>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{user.email}</td>
                  <td className="px-4 py-3 text-xs font-medium">{user.scans}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{user.joinDate}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{user.lastActive}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1.5">
                      <div className={cn("w-2 h-2 rounded-full", user.status === "online" ? "bg-success" : "bg-muted-foreground/40")} />
                      <span className="text-xs capitalize text-muted-foreground">{user.status}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <button className="p-1 rounded-md hover:bg-muted transition-colors">
                      <MoreHorizontal className="w-4 h-4 text-muted-foreground" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// AI Training Page
// ============================================================
export function AdminAITrainingPage() {
  const [autoRetrain, setAutoRetrain] = useState(true);
  const [retrainThreshold, setRetrainThreshold] = useState(50);
  const [useOverridesForTraining, setUseOverridesForTraining] = useState(true);
  const [aiAgentEnabled, setAiAgentEnabled] = useState(true);
  const [agentSuggestions, setAgentSuggestions] = useState(true);
  const [agentAutoReport, setAgentAutoReport] = useState(false);
  const [agentConfidenceThreshold, setAgentConfidenceThreshold] = useState(85);

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">AI Training & Agent</h1>
        <p className="text-xs text-muted-foreground mt-0.5">Configure AI behavior, training pipeline, and diagnostic agent</p>
      </motion.div>

      {/* Agent Config */}
      <motion.div variants={itemVariants} className="card-clinical">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-4 h-4 text-primary" />
          <p className="text-sm font-medium">AI Diagnostic Agent</p>
        </div>
        <p className="text-xs text-muted-foreground mb-4">
          The AI agent uses diagnostic history, overrides, and clinical notes to provide personalized suggestions.
        </p>
        <div className="space-y-4">
          {[
            { label: "Enable AI Agent", desc: "Agent learns from diagnostic decisions to improve suggestions", val: aiAgentEnabled, set: setAiAgentEnabled },
            { label: "Proactive Suggestions", desc: "Show AI suggestions during diagnosis based on patient context", val: agentSuggestions, set: setAgentSuggestions },
            { label: "Auto-Generate Draft Reports", desc: "Create draft clinical reports after diagnosis confirmation", val: agentAutoReport, set: setAgentAutoReport },
          ].map(item => (
            <div key={item.label} className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="text-sm font-medium">{item.label}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{item.desc}</p>
              </div>
              <Toggle value={item.val} onChange={item.set} />
            </div>
          ))}
          <div className="p-3 rounded-lg border">
            <div className="flex items-center justify-between mb-2">
              <div>
                <p className="text-sm font-medium">Agent Confidence Threshold</p>
                <p className="text-xs text-muted-foreground mt-0.5">Only show suggestions above this confidence</p>
              </div>
              <span className="text-mono text-sm font-medium">{agentConfidenceThreshold}%</span>
            </div>
            <input type="range" min="50" max="99" value={agentConfidenceThreshold} onChange={e => setAgentConfidenceThreshold(parseInt(e.target.value))} className="w-full accent-primary h-1.5" />
            <div className="flex justify-between mt-1">
              <span className="text-[10px] text-muted-foreground">More suggestions</span>
              <span className="text-[10px] text-muted-foreground">Higher accuracy</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Training Stats */}
      <motion.div variants={itemVariants} className="card-clinical">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-4 h-4 text-primary" />
          <p className="text-sm font-medium">Training Data Collection</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
          {[
            { label: "Doctor Overrides", value: trainingStats.doctorOverrides },
            { label: "Labeled Images", value: trainingStats.labeledImages },
            { label: "Training Epochs", value: trainingStats.currentEpoch },
          ].map(s => (
            <div key={s.label} className="p-3 rounded-lg bg-primary-muted text-center">
              <p className="text-lg font-medium text-primary">{s.value}</p>
              <p className="text-[10px] text-muted-foreground">{s.label}</p>
            </div>
          ))}
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="text-sm font-medium">Use Overrides for Retraining</p>
              <p className="text-xs text-muted-foreground mt-0.5">Include doctor-overridden grades in training data</p>
            </div>
            <Toggle value={useOverridesForTraining} onChange={setUseOverridesForTraining} />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="text-sm font-medium">Automatic Retraining</p>
              <p className="text-xs text-muted-foreground mt-0.5">Trigger retraining when enough new data is collected</p>
            </div>
            <Toggle value={autoRetrain} onChange={setAutoRetrain} />
          </div>
          {autoRetrain && (
            <div className="p-3 rounded-lg border">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium">Retrain after new samples</p>
                <span className="text-mono text-sm">{retrainThreshold}</span>
              </div>
              <input type="range" min="10" max="200" step="10" value={retrainThreshold} onChange={e => setRetrainThreshold(parseInt(e.target.value))} className="w-full accent-primary h-1.5" />
            </div>
          )}
        </div>
      </motion.div>

      {/* Overrides Queue */}
      <motion.div variants={itemVariants} className="card-clinical">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-warning" />
            <p className="text-sm font-medium">Doctor Overrides Queue</p>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-warning/10 text-warning font-medium">
            {recentOverrides.filter(o => o.status === "pending").length} pending
          </span>
        </div>
        <div className="space-y-2">
          {recentOverrides.map((o, i) => (
            <div key={i} className="flex items-center gap-3 p-3 rounded-lg border flex-wrap">
              <span className="text-mono text-xs text-primary w-16">{o.id}</span>
              <span className="text-xs text-muted-foreground w-20">{o.date}</span>
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">AI: G{o.aiGrade}</span>
                <ChevronRight className="w-3 h-3 text-muted-foreground" />
                <span className="text-xs font-medium">Dr: G{o.doctorGrade}</span>
              </div>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{o.modality}</span>
              <span className="text-[10px] text-muted-foreground">{o.doctor}</span>
              <div className="ml-auto">
                <span className={cn("text-[10px] px-2 py-0.5 rounded-full font-medium",
                  o.status === "approved" ? "bg-success/10 text-success" : "bg-warning/10 text-warning"
                )}>{o.status}</span>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// Datasets Page
// ============================================================
export function AdminDatasetsPage() {
  const [autoLabel, setAutoLabel] = useState(true);
  const [augmentData, setAugmentData] = useState(true);
  const [dataValidation, setDataValidation] = useState(true);

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Labeled Datasets</h1>
        <p className="text-xs text-muted-foreground mt-0.5">Manage training datasets for X-ray and MRI models</p>
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {[
          { label: "Total Images", value: trainingStats.totalImages, icon: FileImage },
          { label: "X-ray", value: trainingStats.xrayImages, icon: Image },
          { label: "MRI", value: trainingStats.mriImages, icon: Layers },
          { label: "Labeled", value: trainingStats.labeledImages, icon: Check },
          { label: "Unlabeled", value: trainingStats.unlabeledImages, icon: Clock },
        ].map(s => (
          <div key={s.label} className="p-3 rounded-lg bg-primary-muted text-center card-clinical">
            <s.icon className="w-4 h-4 text-primary mx-auto mb-1" />
            <p className="text-lg font-medium">{s.value}</p>
            <p className="text-[10px] text-muted-foreground">{s.label}</p>
          </div>
        ))}
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="section-header mb-3">Grade Distribution by Modality</p>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left px-3 py-2 section-header">KL Grade</th>
                <th className="text-left px-3 py-2 section-header">Label</th>
                <th className="text-right px-3 py-2 section-header">X-ray</th>
                <th className="text-right px-3 py-2 section-header">MRI</th>
                <th className="text-right px-3 py-2 section-header">Total</th>
                <th className="text-left px-3 py-2 section-header">Distribution</th>
              </tr>
            </thead>
            <tbody>
              {datasetBreakdown.map(row => (
                <tr key={row.grade} className="border-b last:border-0">
                  <td className="px-3 py-2.5 text-sm font-medium">Grade {row.grade}</td>
                  <td className="px-3 py-2.5 text-xs text-muted-foreground">{row.label}</td>
                  <td className="px-3 py-2.5 text-mono text-sm text-right">{row.xray}</td>
                  <td className="px-3 py-2.5 text-mono text-sm text-right">{row.mri}</td>
                  <td className="px-3 py-2.5 text-mono text-sm font-medium text-right">{row.total}</td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-primary rounded-full" style={{ width: `${(row.total / trainingStats.totalImages) * 100}%` }} />
                      </div>
                      <span className="text-mono text-[10px] text-muted-foreground w-8">{Math.round((row.total / trainingStats.totalImages) * 100)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="text-sm font-medium mb-4">Dataset Management</p>
        <div className="space-y-3">
          {[
            { label: "Auto-Label Confirmed Diagnoses", desc: "Automatically add confirmed images to labeled dataset", val: autoLabel, set: setAutoLabel },
            { label: "Data Augmentation", desc: "Apply rotation, flipping, and contrast augmentation", val: augmentData, set: setAugmentData },
            { label: "Quality Validation", desc: "Run quality checks before adding to dataset", val: dataValidation, set: setDataValidation },
          ].map(item => (
            <div key={item.label} className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="text-sm font-medium">{item.label}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{item.desc}</p>
              </div>
              <Toggle value={item.val} onChange={item.set} />
            </div>
          ))}
        </div>
        <div className="flex gap-2 mt-4 flex-wrap">
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
            <Upload className="w-4 h-4" />Import Dataset
          </button>
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
            <Download className="w-4 h-4" />Export
          </button>
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors">
            <Eye className="w-4 h-4" />Browse
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// Models Page
// ============================================================
export function AdminModelsPage() {
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Model Management</h1>
        <p className="text-xs text-muted-foreground mt-0.5">Active architectures, version history, and training configuration</p>
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="section-header mb-3">X-Ray Models (Phase I — Ensemble + Majority Voting)</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
          {[
            { name: "DenseNet201", task: "Detailed Classification", accuracy: "94.2%", status: "active" },
            { name: "ViT-B/16", task: "Global Context Analysis", accuracy: "92.8%", status: "active" },
            { name: "ResNet50", task: "Baseline Comparison", accuracy: "89.5%", status: "standby" },
          ].map(m => (
            <div key={m.name} className={cn("p-4 rounded-lg border", m.status === "active" && "border-primary/30 bg-primary-muted/30")}>
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium">{m.name}</p>
                <span className={cn("text-[10px] px-1.5 py-0.5 rounded-full font-medium", m.status === "active" ? "bg-success/10 text-success" : "bg-muted text-muted-foreground")}>{m.status}</span>
              </div>
              <p className="text-xs text-muted-foreground">{m.task}</p>
              <p className="text-xs mt-1">Accuracy: <span className="font-medium">{m.accuracy}</span></p>
            </div>
          ))}
        </div>
        <p className="section-header mb-3">MRI Models (Phase II — Swin-UNet + Classifier)</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            { name: "Swin-UNet + DenseNet201", task: "Artifact Removal + Classification", accuracy: "93.5%", status: "active" },
            { name: "Swin-UNet + ViT-B/16", task: "Artifact Removal + Global Analysis", accuracy: "91.7%", status: "active" },
            { name: "Swin-UNet + ResNet50", task: "Artifact Removal + Baseline", accuracy: "88.2%", status: "standby" },
          ].map(m => (
            <div key={m.name} className={cn("p-4 rounded-lg border", m.status === "active" && "border-primary/30 bg-primary-muted/30")}>
              <div className="flex items-center justify-between mb-2">
                <p className="text-[11px] font-medium">{m.name}</p>
                <span className={cn("text-[10px] px-1.5 py-0.5 rounded-full font-medium", m.status === "active" ? "bg-success/10 text-success" : "bg-muted text-muted-foreground")}>{m.status}</span>
              </div>
              <p className="text-xs text-muted-foreground">{m.task}</p>
              <p className="text-xs mt-1">Accuracy: <span className="font-medium">{m.accuracy}</span></p>
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="text-sm font-medium mb-4">Version History</p>
        <div className="space-y-2">
          {modelVersions.map(v => (
            <div key={v.version} className={cn("flex items-center gap-4 p-3 rounded-lg border flex-wrap", v.status === "active" && "border-primary/30 bg-primary-muted/20")}>
              <span className="text-mono text-sm font-medium w-16">{v.version}</span>
              <span className="text-xs text-muted-foreground w-24">{v.date}</span>
              <span className="text-xs w-20">Acc: <span className="font-medium">{v.accuracy}</span></span>
              <span className="text-xs text-muted-foreground flex-1">{v.dataset}</span>
              <span className={cn("text-[10px] px-2 py-0.5 rounded-full font-medium", v.status === "active" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>{v.status}</span>
              {v.status === "archived" && <button className="text-xs text-primary hover:text-primary/80">Restore</button>}
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="text-sm font-medium mb-3">Training Configuration</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {[
            { label: "Learning Rate", value: "0.0001" },
            { label: "Batch Size", value: "32" },
            { label: "Epochs", value: "150" },
            { label: "Image Size", value: "224x224" },
            { label: "Validation Split", value: "20%" },
          ].map(f => (
            <div key={f.label}>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">{f.label}</label>
              <input defaultValue={f.value} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
            </div>
          ))}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Optimizer</label>
            <select defaultValue="adam" className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20">
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="adamw">AdamW</option>
            </select>
          </div>
        </div>
        <button className="mt-4 flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
          <RefreshCw className="w-4 h-4" />Start Training Run
        </button>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// MRI Pipeline Page
// ============================================================
export function AdminMRIPipelinePage() {
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">MRI Pipeline (Phase II)</h1>
        <p className="text-xs text-muted-foreground mt-0.5">Swin-UNet artifact removal and enhancement pipeline configuration</p>
      </motion.div>

      <motion.div variants={itemVariants} className="space-y-4">
        {[
          { stage: 1, title: "Pre-training: Image Restoration", status: "Trained", desc: "Swin-UNet trained on KMAR-50K to learn feature reconstruction.", stats: [{ l: "Architecture", v: "Swin-UNet" }, { l: "Training Data", v: "KMAR-50K" }, { l: "Samples", v: "50,000" }] },
          { stage: 2, title: "Enhancement: Artifact Removal", status: "Active", desc: "Raw SKM-TEA images cleaned via Swin-UNet.", stats: [{ l: "Input", v: "Raw SKM-TEA" }, { l: "Output", v: "Cleaned MRI" }, { l: "Quality", v: "88.3%" }, { l: "Processed", v: "744" }] },
          { stage: 3, title: "Downstream: Classification", status: "Active", desc: "Cleaned data trains high-precision classifiers.", stats: [{ l: "Classifier", v: "DenseNet201" }, { l: "Accuracy", v: "93.5%" }, { l: "Improvement", v: "+6.2%" }] },
        ].map(s => (
          <div key={s.stage} className="card-clinical border-primary/20">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-[10px] font-bold">{s.stage}</div>
              <p className="text-sm font-medium">{s.title}</p>
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-success/10 text-success font-medium ml-auto">{s.status}</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">{s.desc}</p>
            <div className={cn("grid gap-2", s.stats.length <= 3 ? "grid-cols-3" : "grid-cols-2 sm:grid-cols-4")}>
              {s.stats.map(st => (
                <div key={st.l} className="p-2 rounded bg-background border text-center">
                  <p className="text-[10px] text-muted-foreground">{st.l}</p>
                  <p className="text-xs font-medium">{st.v}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="text-sm font-medium mb-3">Pipeline Configuration</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {[
            { label: "Restoration Model", type: "select", options: ["Swin-UNet", "Standard U-Net", "Restormer"] },
            { label: "Pre-training Dataset", type: "select", options: ["KMAR-50K (Recommended)", "Custom"] },
            { label: "Target Dataset", type: "select", options: ["SKM-TEA", "fastMRI", "Custom"] },
            { label: "Quality Threshold", type: "input", value: "75" },
            { label: "Artifact Types", type: "select", options: ["All (Motion + Stripe)", "Motion Blur", "Stripe Artifacts"] },
            { label: "Swin Window Size", type: "input", value: "8" },
          ].map(f => (
            <div key={f.label}>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">{f.label}</label>
              {f.type === "select" ? (
                <select defaultValue={f.options?.[0]} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20">
                  {f.options?.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              ) : (
                <input defaultValue={f.value} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
              )}
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="card-clinical">
        <p className="text-sm font-medium mb-4">Research Contributions</p>
        <div className="space-y-3">
          {[
            { title: "Cross-Dataset Synergy", desc: "KMAR-50K data improves SKM-TEA diagnostic performance." },
            { title: "Clinical Transparency", desc: "Ensemble Learning + Grad-CAM builds physician trust." },
            { title: "Dual-Modality Integration", desc: "Bone-level X-ray analysis (Phase I) + soft-tissue MRI (Phase II)." },
          ].map(item => (
            <div key={item.title} className="p-3 rounded-lg border">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-primary" />
                <p className="text-sm font-medium">{item.title}</p>
              </div>
              <p className="text-xs text-muted-foreground">{item.desc}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// System Page
// ============================================================
export function AdminSystemPage() {
  const [logFilter, setLogFilter] = useState("all");
  const filteredLogs = logFilter === "all" ? systemLogs : systemLogs.filter(l => l.type === logFilter);

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">System</h1>
        <p className="text-xs text-muted-foreground mt-0.5">System health, logs, and infrastructure monitoring</p>
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "CPU Usage", value: "34%", icon: Monitor, color: "text-success" },
          { label: "Memory", value: "4.2 / 8 GB", icon: HardDrive, color: "text-primary" },
          { label: "Storage", value: "12.8 / 50 GB", icon: Database, color: "text-primary" },
          { label: "GPU (Inference)", value: "62%", icon: Zap, color: "text-warning" },
        ].map(item => (
          <div key={item.label} className="card-clinical">
            <div className="flex items-center gap-2 mb-2">
              <item.icon className={cn("w-4 h-4", item.color)} />
              <span className="text-xs text-muted-foreground">{item.label}</span>
            </div>
            <p className="text-lg font-semibold">{item.value}</p>
          </div>
        ))}
      </motion.div>

      {/* System Info */}
      <motion.div variants={itemVariants} className="card-clinical">
        <p className="text-sm font-medium mb-4">System Information</p>
        <div className="space-y-2">
          {[
            { label: "App Version", value: "KneeXpert v1.4.0" },
            { label: "AI Engine", value: `Model ${trainingStats.modelVersion}` },
            { label: "Last Training", value: trainingStats.lastTrainingDate },
            { label: "Database", value: `${trainingStats.totalImages} images` },
            { label: "Storage Used", value: "12.4 GB / 50 GB" },
            { label: "API Status", value: "Operational" },
          ].map(item => (
            <div key={item.label} className="flex items-center justify-between p-3 rounded-lg border">
              <p className="text-sm text-muted-foreground">{item.label}</p>
              <p className="text-sm font-medium">{item.value}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Logs */}
      <motion.div variants={itemVariants} className="card-clinical">
        <div className="flex items-center justify-between mb-3">
          <p className="section-header">Activity Log</p>
          <div className="flex gap-0.5 bg-muted rounded-lg p-0.5">
            {["all", "info", "success", "warning"].map(f => (
              <button key={f} onClick={() => setLogFilter(f)} className={cn("px-2 py-1 rounded-md text-[10px] font-medium capitalize", logFilter === f ? "bg-background shadow-sm text-foreground" : "text-muted-foreground")}>
                {f}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          {filteredLogs.map((log, i) => (
            <div key={i} className="flex items-start gap-3 py-2 border-b last:border-0">
              <div className={cn("w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 mt-0.5",
                log.type === "success" ? "bg-success/10" : log.type === "warning" ? "bg-warning/10" : "bg-primary-muted"
              )}>
                {log.type === "success" ? <CheckCircle2 className="w-3.5 h-3.5 text-success" /> :
                 log.type === "warning" ? <AlertTriangle className="w-3.5 h-3.5 text-warning" /> :
                 <Activity className="w-3.5 h-3.5 text-primary" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium">{log.event}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <p className="text-[10px] text-muted-foreground">{log.time} · Today</p>
                  <span className="text-[10px] text-muted-foreground/60">{log.detail}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

// Default export for backward compat
export default function AdminPage() {
  return <AdminOverviewPage />;
}
