import { useMemo } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ClipboardCheck, FileText, GitCompare, Pencil } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { usePatients } from "@/context/PatientContext";
import { buildAuditSummary, percent, type GradeAgreement, type GradeSource } from "@/lib/audit";

const containerVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] as const, staggerChildren: 0.05 } },
};
const itemVariants = {
  hidden: { opacity: 0, y: 4 },
  visible: { opacity: 1, y: 0 },
};

const SOURCE_LABEL: Record<GradeSource, string> = {
  xray: "X-ray",
  mri: "MRI",
  combined: "Combined (fused)",
};

function pct(part: number, whole: number): string {
  const value = percent(part, whole);
  return value === null ? "—" : `${value}%`;
}

function AgreementRow({ source, a }: { source: GradeSource; a: GradeAgreement }) {
  return (
    <tr className="border-t">
      <td className="py-2 pr-3 text-xs font-medium">{SOURCE_LABEL[source]}</td>
      <td className="py-2 px-3 text-xs text-right tabular-nums">{a.reviewed}</td>
      <td className="py-2 px-3 text-xs text-right tabular-nums">{pct(a.exact, a.reviewed)}</td>
      <td className="py-2 px-3 text-xs text-right tabular-nums">{pct(a.withinOne, a.reviewed)}</td>
      <td className="py-2 px-3 text-xs text-right tabular-nums">{pct(a.overridden, a.reviewed)}</td>
      <td className="py-2 pl-3 text-xs text-right tabular-nums">
        {a.meanAbsoluteDifference === null ? "—" : a.meanAbsoluteDifference.toFixed(2)}
      </td>
    </tr>
  );
}

/**
 * Performance audit (poster step 10). Read-only: computed from the
 * confirmations, overrides and note sign-offs already in patient records, and
 * fed back into nothing.
 */
export default function AdminAuditPage() {
  const { patients } = usePatients();
  const summary = useMemo(() => buildAuditSummary(patients), [patients]);
  const { agreement, notes, discordance } = summary;

  const kpis = [
    { label: "Reviewed reports", value: String(agreement.reviewed), icon: ClipboardCheck },
    { label: "AI = final grade", value: pct(agreement.exact, agreement.reviewed), icon: CheckCircle2 },
    { label: "Within one grade", value: pct(agreement.withinOne, agreement.reviewed), icon: GitCompare },
    { label: "Override rate", value: pct(agreement.overridden, agreement.reviewed), icon: Pencil },
    { label: "Notes signed off", value: String(notes.accepted + notes.edited), icon: FileText },
    { label: "Notes rejected", value: pct(notes.rejected, notes.total), icon: AlertTriangle },
  ];

  const noteChart = (["xray", "mri", "combined"] as const).map(m => ({
    modality: SOURCE_LABEL[m],
    Accepted: summary.notesByModality[m].accepted,
    Edited: summary.notesByModality[m].edited,
    Rejected: summary.notesByModality[m].rejected,
  }));

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 space-y-5">
      <motion.div variants={itemVariants}>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Performance Audit</h1>
        <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
          How the AI's output fared under clinician review
        </p>
      </motion.div>

      <motion.div
        variants={itemVariants}
        className="p-3 rounded-lg border border-warning/20 bg-warning/5 flex items-start gap-2"
      >
        <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
        <p className="text-[11px] text-warning leading-relaxed">
          Read-only. Computed from confirmations, overrides and note sign-offs stored in this browser,
          including the seeded demo patients. Nothing here changes or retrains the models.
        </p>
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {kpis.map(stat => (
          <div key={stat.label} className="card-clinical">
            <div className="w-8 h-8 rounded-lg bg-primary-muted flex items-center justify-center mb-2">
              <stat.icon className="w-4 h-4 text-primary" />
            </div>
            <p className="text-lg sm:text-xl font-semibold tabular-nums">{stat.value}</p>
            <p className="text-[10px] sm:text-[11px] text-muted-foreground">{stat.label}</p>
          </div>
        ))}
      </motion.div>

      <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card-clinical">
          <p className="section-header mb-3">AI grade vs final grade</p>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  <th className="pb-2 pr-3 text-left font-medium">Grade from</th>
                  <th className="pb-2 px-3 text-right font-medium">Reviewed</th>
                  <th className="pb-2 px-3 text-right font-medium">Exact</th>
                  <th className="pb-2 px-3 text-right font-medium">±1</th>
                  <th className="pb-2 px-3 text-right font-medium">Overridden</th>
                  <th className="pb-2 pl-3 text-right font-medium">Mean |Δ|</th>
                </tr>
              </thead>
              <tbody>
                {(["xray", "mri", "combined"] as const).map(source => (
                  <AgreementRow key={source} source={source} a={summary.agreementBySource[source]} />
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-muted-foreground mt-3 leading-relaxed">
            A combined grade is the orchestrator's confidence-weighted fusion of an X-ray and an MRI.
            {discordance.fused > 0 &&
              ` ${discordance.discordant} of ${discordance.fused} fused reports had disagreeing studies` +
                (discordance.meanGap === null ? "." : ` (mean gap ${discordance.meanGap.toFixed(1)} grades).`)}
          </p>
        </div>

        <div className="card-clinical">
          <p className="section-header mb-3">Clinical note outcomes</p>
          {notes.total === 0 ? (
            <p className="text-xs text-muted-foreground py-8 text-center">
              No clinical notes have been signed off or rejected yet.
            </p>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={noteChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" />
                  <XAxis dataKey="modality" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={24} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid hsl(220, 13%, 91%)" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Accepted" stackId="n" fill="hsl(160, 84%, 39%)" />
                  <Bar dataKey="Edited" stackId="n" fill="hsl(217, 91%, 60%)" />
                  <Bar dataKey="Rejected" stackId="n" fill="hsl(0, 72%, 51%)" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-[10px] text-muted-foreground mt-2">
                {notes.meanGateAttempts === null
                  ? ""
                  : `Mean generation attempts per reviewed note: ${notes.meanGateAttempts.toFixed(1)}. `}
                Withheld notes are never signed off, so they do not appear here.
              </p>
            </>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}
