import { useId, useState } from "react";
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

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { groundingBand } from "@/lib/clinicalNote";
import { cn } from "@/lib/utils";
import type {
  ClinicalNoteFallbackReason,
  ClinicalNoteResponse,
  NoteSentence,
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

/**
 * Panel state as a discriminated union, so illegal combinations — an error with
 * a response, a ready state with neither — cannot be constructed.
 */
export type ClinicalNoteViewState =
  | { kind: "loading" }
  | { kind: "ready"; response: ClinicalNoteResponse }
  | { kind: "error"; error: Error };

type Props = {
  state: ClinicalNoteViewState;
  /** Verdict already recorded for this modality, if any. */
  status: NoteReviewStatus | null;
  reviewedText?: string;
  onAccept: () => void;
  onEdit: (text: string) => void;
  onReject: (reason: string) => void;
  onRegenerate: () => void;
};

const FALLBACK_COPY: Record<ClinicalNoteFallbackReason, string> = {
  gate_rejected: "AI note unavailable — the draft could not be grounded.",
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

const SENTENCE_LABEL_COPY = {
  entailment: "supported",
  neutral: "not found in evidence",
  contradiction: "contradicts evidence",
} as const;

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** Shared alert shell, so the three tones differ only in colour and icon. */
function NoteAlert({
  tone,
  children,
  testId,
}: {
  tone: "warning" | "destructive";
  children: React.ReactNode;
  testId?: string;
}) {
  return (
    <div
      data-testid={testId}
      className={cn(
        "p-2 rounded-lg border flex items-start gap-2",
        tone === "warning"
          ? "bg-warning/5 border-warning/20"
          : "bg-destructive/5 border-destructive/20",
      )}
    >
      <AlertTriangle
        className={cn(
          "w-3.5 h-3.5 flex-shrink-0 mt-0.5",
          tone === "warning" ? "text-warning" : "text-destructive",
        )}
      />
      <div
        className={cn(
          "text-[11px] leading-relaxed",
          tone === "warning" ? "text-warning" : "text-destructive",
        )}
      >
        {children}
      </div>
    </div>
  );
}

/** Per-sentence verdict. The citation shown is the entailed passage, not the
 *  marker the model wrote, so it cannot be fabricated. */
function SentenceVerdicts({
  sentences,
  passageNumber,
}: {
  sentences: NoteSentence[];
  passageNumber: (passageId: string | null | undefined) => number | null;
}) {
  if (!sentences.length) return null;
  return (
    <div className="space-y-1">
      <p className="text-[11px] font-medium text-muted-foreground">
        Per-claim support
      </p>
      {sentences.map(sentence => {
        const number = passageNumber(sentence.supporting_passage_id);
        return (
          <div
            key={sentence.text}
            className="text-[10px] leading-relaxed flex items-start gap-1.5"
          >
            <span
              className={cn(
                "px-1 rounded flex-shrink-0 font-medium",
                sentence.label === "entailment"
                  ? "bg-success/10 text-success"
                  : sentence.label === "contradiction"
                    ? "bg-destructive/10 text-destructive"
                    : "bg-muted text-muted-foreground",
              )}
            >
              {SENTENCE_LABEL_COPY[sentence.label]}
            </span>
            <span className="text-muted-foreground">
              {number === null ? "(no source)" : `[${number}]`} {sentence.text}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function ClinicalNotePanel({
  state,
  status,
  reviewedText,
  onAccept,
  onEdit,
  onReject,
  onRegenerate,
}: Props) {
  const [draft, setDraft] = useState<string | null>(null);
  const [rejecting, setRejecting] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [showEvidence, setShowEvidence] = useState(false);
  const evidenceId = useId();

  const response = state.kind === "ready" ? state.response : null;
  const note = response?.clinical_note ?? null;
  const band = groundingBand(
    response?.grounding_score ?? null,
    response?.gate_threshold,
  );
  const signedOff = status === "accepted" || status === "edited";

  const passageNumber = (passageId: string | null | undefined) => {
    if (!passageId || !response) return null;
    const index = response.retrieved_passages.findIndex(
      passage => passage.passage_id === passageId,
    );
    return index === -1 ? null : index + 1;
  };

  return (
    <section className="p-4 rounded-xl border bg-card" aria-label="Guideline-grounded note">
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <FileText className="w-4 h-4 text-primary" aria-hidden="true" />
        <h3 className="text-sm font-medium">Guideline-Grounded Note</h3>
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

      {state.kind === "loading" && (
        <div className="space-y-2" data-testid="clinical-note-loading" role="status">
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground mb-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
            Drafting from retrieved clinical evidence…
          </div>
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-[92%]" />
          <Skeleton className="h-3 w-[78%]" />
        </div>
      )}

      {state.kind === "error" && (
        <NoteAlert tone="destructive">
          {state.error.message || "Could not reach the clinical note service."}
        </NoteAlert>
      )}

      {/* Withheld: say so plainly, and say why. Never substitute other text. */}
      {state.kind === "ready" && !note && (
        <NoteAlert tone="warning" testId="clinical-note-unavailable">
          <p>
            {response?.fallback_reason
              ? FALLBACK_COPY[response.fallback_reason]
              : "AI note unavailable."}
          </p>
          {response?.fallback_reason === "gate_rejected" && (
            <p className="mt-1 opacity-80">
              {plural(response.gate_attempts, "attempt")} below the groundedness
              threshold. The deterministic interpretation above is unaffected.
            </p>
          )}
        </NoteAlert>
      )}

      {state.kind === "ready" && note && (
        <div className="space-y-3" data-testid="clinical-note-body">
          <NoteAlert tone="warning">
            Drafted by a language model from retrieved literature. Requires
            physician review before it enters the patient record.
          </NoteAlert>

          {draft === null ? (
            <p className="text-xs leading-relaxed text-foreground/90">
              {reviewedText?.trim() ? reviewedText : note.synthesis}
            </p>
          ) : (
            <Textarea
              value={draft}
              onChange={event => setDraft(event.target.value)}
              rows={6}
              aria-label="Edit clinical note"
              className="text-xs leading-relaxed"
            />
          )}

          {/* After an edit, both versions stay visible: "reviewed by" is only
              meaningful if what was changed can be seen. */}
          {status === "edited" && reviewedText?.trim() && draft === null && (
            <details className="text-[10px]">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                Show the model's original wording
              </summary>
              <p className="mt-1 p-2 rounded-lg bg-muted/40 border text-muted-foreground leading-relaxed">
                {note.synthesis}
              </p>
            </details>
          )}

          {note.next_steps.length > 0 && (
            <div>
              <p className="text-[11px] font-medium text-muted-foreground mb-1">
                Suggested next steps
              </p>
              <ul className="space-y-1">
                {note.next_steps.map(step => (
                  <li
                    key={step}
                    className="text-[11px] leading-relaxed flex gap-1.5"
                  >
                    <span className="text-primary" aria-hidden="true">
                      •
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowEvidence(open => !open)}
            aria-expanded={showEvidence}
            aria-controls={evidenceId}
            className="h-auto px-0 py-0 text-[11px] text-muted-foreground hover:bg-transparent hover:text-foreground"
          >
            <ChevronDown
              className={cn(
                "w-3 h-3 mr-1 transition-transform",
                showEvidence && "rotate-180",
              )}
              aria-hidden="true"
            />
            {showEvidence ? "Hide" : "Show"} evidence (
            {response?.retrieved_passages.length ?? 0})
          </Button>

          {showEvidence && (
            <div id={evidenceId} className="space-y-2" data-testid="clinical-note-evidence">
              {/* The raw figure lives here rather than in the header: the
                  threshold is provisional, so a bare number should not read as
                  a calibrated score at a glance. */}
              <p className="text-[10px] text-muted-foreground">
                Groundedness {response?.grounding_score?.toFixed(3) ?? "n/a"} vs
                threshold {response?.gate_threshold?.toFixed(2) ?? "n/a"} ·{" "}
                {plural(response?.gate_attempts ?? 0, "attempt")} ·{" "}
                {response?.generated_by}
              </p>

              <SentenceVerdicts
                sentences={note.sentences}
                passageNumber={passageNumber}
              />

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
                    <span className="px-1 rounded bg-background text-muted-foreground">
                      {passage.licence}
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
            <div className="space-y-2 pt-1">
              {rejecting && (
                <div className="space-y-2">
                  <Textarea
                    value={rejectReason}
                    onChange={event => setRejectReason(event.target.value)}
                    rows={2}
                    placeholder="Why is this note being rejected?"
                    aria-label="Reason for rejecting the note"
                    className="text-xs"
                  />
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="destructive"
                      disabled={!rejectReason.trim()}
                      onClick={() => {
                        onReject(rejectReason.trim());
                        setRejecting(false);
                      }}
                    >
                      Confirm rejection
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() => setRejecting(false)}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {!rejecting && draft === null && (
                <div className="flex items-center gap-2 flex-wrap">
                  <Button type="button" size="sm" variant="outline" onClick={onAccept}>
                    <Check className="w-3 h-3 mr-1.5" aria-hidden="true" /> Accept
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={() => setDraft(note.synthesis)}
                  >
                    <Pencil className="w-3 h-3 mr-1.5" aria-hidden="true" /> Edit
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => setRejecting(true)}
                  >
                    <X className="w-3 h-3 mr-1.5" aria-hidden="true" /> Reject
                  </Button>
                  <Button type="button" size="sm" variant="ghost" onClick={onRegenerate}>
                    <RefreshCw className="w-3 h-3 mr-1.5" aria-hidden="true" />{" "}
                    Regenerate
                  </Button>
                </div>
              )}

              {!rejecting && draft !== null && (
                <div className="flex items-center gap-2 flex-wrap">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      onEdit(draft);
                      setDraft(null);
                    }}
                  >
                    <Check className="w-3 h-3 mr-1.5" aria-hidden="true" /> Save &amp;
                    sign off
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={() => setDraft(null)}
                  >
                    Cancel
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
