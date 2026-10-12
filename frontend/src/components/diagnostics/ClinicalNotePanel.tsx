import { useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  FileText,
  Loader2,
  Pencil,
  RefreshCw,
  X,
} from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import { groundingBand } from "@/lib/clinicalNote";
import { cn } from "@/lib/utils";
import type {
  ClinicalNoteFallbackReason,
  ClinicalNoteResponse,
} from "@/lib/diagnosticApi";
import type { NoteReviewStatus } from "@/data/patients";

/**
 * The AI-generated clinical note, held deliberately apart from the
 * deterministic interpretation beside it.
 *
 * Two rules this component exists to enforce:
 *
 * 1. It never renders deterministic `FeedbackBundle` text. When no note was
 *    served it says so and says why. Filling the gap with dictionary text would
 *    misreport provenance, and it is the one thing the whole grounding pipeline
 *    is measured against.
 * 2. Nothing here enters the patient record until a clinician accepts or edits
 *    it, which is what the sign-off row below is for.
 */

type Props = {
  state: "loading" | "ready" | "error";
  response: ClinicalNoteResponse | null;
  error: Error | null;
  /** Verdict already recorded for this modality, if any. */
  status: NoteReviewStatus | null;
  reviewedText?: string;
  onAccept: () => void;
  onEdit: (text: string) => void;
  onReject: (reason: string) => void;
  onRegenerate: () => void;
};

const FALLBACK_COPY: Record<ClinicalNoteFallbackReason, string> = {
  gate_rejected:
    "AI note unavailable — the draft could not be grounded in the retrieved evidence after the retry budget.",
  llm_unavailable:
    "AI note unavailable — the language model service is not reachable.",
  timeout: "AI note unavailable — generation timed out.",
  parse_failed:
    "AI note unavailable — the model did not return readable output.",
  empty_corpus:
    "AI note unavailable — no supporting evidence was retrieved for this case.",
};

const BAND_COPY = {
  supported: { label: "Evidence-supported", tone: "bg-success/10 text-success" },
  partial: { label: "Partially supported", tone: "bg-warning/10 text-warning" },
  weak: { label: "Weakly supported", tone: "bg-destructive/10 text-destructive" },
  unknown: { label: "Not scored", tone: "bg-muted text-muted-foreground" },
} as const;

export function ClinicalNotePanel({
  state,
  response,
  error,
  status,
  reviewedText,
  onAccept,
  onEdit,
  onReject,
  onRegenerate,
}: Props) {
  const [draft, setDraft] = useState<string | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);

  const note = response?.clinical_note ?? null;
  const band = groundingBand(
    response?.grounding_score ?? null,
    response?.gate_threshold,
  );
  const signedOff = status === "accepted" || status === "edited";

  return (
    <div className="p-4 rounded-xl border bg-card">
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <FileText className="w-4 h-4 text-primary" />
        <p className="text-sm font-medium">Guideline-Grounded Note</p>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
          AI-GENERATED
        </span>
        {note && (
          <span
            className={cn(
              "text-[10px] px-1.5 py-0.5 rounded font-medium",
              BAND_COPY[band].tone,
            )}
          >
            {BAND_COPY[band].label}
          </span>
        )}
        {signedOff && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">
            {status === "edited" ? "EDITED & SIGNED OFF" : "SIGNED OFF"}
          </span>
        )}
        {status === "rejected" && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-destructive/10 text-destructive font-medium">
            REJECTED
          </span>
        )}
      </div>

      {state === "loading" && (
        <div className="space-y-2" data-testid="clinical-note-loading">
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground mb-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Drafting from retrieved clinical evidence…
          </div>
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-[92%]" />
          <Skeleton className="h-3 w-[78%]" />
        </div>
      )}

      {state === "error" && (
        <div className="p-2 rounded-lg bg-destructive/5 border border-destructive/20 flex items-start gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-destructive flex-shrink-0 mt-0.5" />
          <p className="text-[11px] text-destructive leading-relaxed">
            {error?.message ?? "Could not reach the clinical note service."}
          </p>
        </div>
      )}

      {/* Withheld: say so plainly, and say why. Never substitute other text. */}
      {state === "ready" && !note && (
        <div
          className="p-2 rounded-lg bg-warning/5 border border-warning/20 flex items-start gap-2"
          data-testid="clinical-note-unavailable"
        >
          <AlertTriangle className="w-3.5 h-3.5 text-warning flex-shrink-0 mt-0.5" />
          <div className="text-[11px] text-warning leading-relaxed">
            <p>
              {response?.fallback_reason
                ? FALLBACK_COPY[response.fallback_reason]
                : "AI note unavailable."}
            </p>
            {response?.fallback_reason === "gate_rejected" && (
              <p className="mt-1 opacity-80">
                {response.gate_attempts} attempt
                {response.gate_attempts === 1 ? "" : "s"} fell below the
                groundedness threshold. The deterministic interpretation above is
                unaffected.
              </p>
            )}
          </div>
        </div>
      )}

      {state === "ready" && note && (
        <div className="space-y-3" data-testid="clinical-note-body">
          <div className="p-2 rounded-lg bg-warning/5 border border-warning/20 flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-warning flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-warning leading-relaxed">
              Drafted by a language model from retrieved literature. Requires
              physician review before it enters the patient record.
            </p>
          </div>

          {draft === null ? (
            <p className="text-xs leading-relaxed text-foreground/90">
              {reviewedText?.trim() ? reviewedText : note.synthesis}
            </p>
          ) : (
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              rows={6}
              aria-label="Edit clinical note"
              className="w-full text-xs leading-relaxed p-2 rounded-lg border bg-background"
            />
          )}

          {note.next_steps.length > 0 && (
            <div>
              <p className="text-[11px] font-medium text-muted-foreground mb-1">
                Suggested next steps
              </p>
              <ul className="space-y-1">
                {note.next_steps.map((step) => (
                  <li
                    key={step}
                    className="text-[11px] leading-relaxed flex gap-1.5"
                  >
                    <span className="text-primary">•</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowEvidence((open) => !open)}
            className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <ChevronDown
              className={cn(
                "w-3 h-3 transition-transform",
                showEvidence && "rotate-180",
              )}
            />
            {showEvidence ? "Hide" : "Show"} evidence (
            {response?.retrieved_passages.length ?? 0})
          </button>

          {showEvidence && (
            <div className="space-y-2" data-testid="clinical-note-evidence">
              {/* The raw figure lives here rather than in the header: the
                  threshold is provisional, so a bare number should not read as
                  a calibrated score at a glance. */}
              <p className="text-[10px] text-muted-foreground">
                Groundedness {response?.grounding_score?.toFixed(3) ?? "n/a"} vs
                threshold {response?.gate_threshold?.toFixed(2) ?? "n/a"} ·{" "}
                {response?.gate_attempts} attempt
                {response?.gate_attempts === 1 ? "" : "s"} ·{" "}
                {response?.generated_by}
              </p>
              {response?.retrieved_passages.map((passage, index) => (
                <div
                  key={passage.passage_id}
                  className="p-2 rounded-lg bg-muted/40 border text-[10px] leading-relaxed"
                >
                  <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                    <span className="font-medium">[{index + 1}]</span>
                    <span className="text-muted-foreground">
                      {passage.source}
                    </span>
                    <span className="px-1 rounded bg-background text-muted-foreground">
                      {passage.source_type}
                    </span>
                  </div>
                  <p className="text-muted-foreground">{passage.text}</p>
                  {passage.url && (
                    <a
                      href={passage.url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-primary hover:underline"
                    >
                      Source
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Sign-off. Independent of agreeing with the KL grade: a clinician
              may override the grade and still accept the management advice. */}
          {!signedOff && status !== "rejected" && (
            <div className="flex items-center gap-2 pt-1 flex-wrap">
              {draft === null ? (
                <>
                  <button
                    type="button"
                    onClick={onAccept}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-success/10 text-success text-[11px] font-medium hover:bg-success/20"
                  >
                    <Check className="w-3 h-3" /> Accept
                  </button>
                  <button
                    type="button"
                    onClick={() => setDraft(note.synthesis)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-muted text-[11px] font-medium hover:bg-muted/70"
                  >
                    <Pencil className="w-3 h-3" /> Edit
                  </button>
                  <button
                    type="button"
                    onClick={() => onReject("Rejected by reviewing clinician")}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-destructive/10 text-destructive text-[11px] font-medium hover:bg-destructive/20"
                  >
                    <X className="w-3 h-3" /> Reject
                  </button>
                  <button
                    type="button"
                    onClick={onRegenerate}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] text-muted-foreground hover:text-foreground"
                  >
                    <RefreshCw className="w-3 h-3" /> Regenerate
                  </button>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => {
                      onEdit(draft);
                      setDraft(null);
                    }}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-success/10 text-success text-[11px] font-medium hover:bg-success/20"
                  >
                    <Check className="w-3 h-3" /> Save &amp; sign off
                  </button>
                  <button
                    type="button"
                    onClick={() => setDraft(null)}
                    className="px-3 py-1.5 rounded-lg bg-muted text-[11px] font-medium hover:bg-muted/70"
                  >
                    Cancel
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
