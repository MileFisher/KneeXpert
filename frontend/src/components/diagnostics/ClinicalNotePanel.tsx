import { useId, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ExternalLink,
  Loader2,
  Pencil,
  RefreshCw,
  Stethoscope,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { noteGroundingBand } from "@/lib/clinicalNote";
import { cn } from "@/lib/utils";
import type {
  ClinicalNoteFallbackReason,
  ClinicalNoteResponse,
  NoteSentence,
  RetrievedPassage,
} from "@/lib/diagnosticApi";
import type { NoteReviewStatus } from "@/data/patients";

/**
 * The AI-generated clinical note, held deliberately apart from the
 * deterministic interpretation beside it.
 *
 * Two rules this component exists to enforce:
 *
 * 1. It never renders text the model did not write. When no note was served it
 *    says so and says why — no deterministic `FeedbackBundle` text, no static
 *    reference list. Filling the gap would misreport provenance, and an honest
 *    gap is the one thing the whole grounding pipeline is measured against.
 * 2. Nothing here enters the patient record until a clinician accepts or edits
 *    it, which is what the sign-off row below is for.
 *
 * Citations shown beside the note are the passages the entailment model found
 * supporting each sentence. The model's own `[n]` markers are never surfaced:
 * they can point at the wrong passage or at none, and measuring how often they
 * do is what `citation_agreement_rate` is for.
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
  /** Heading; distinguishes a combined note from a per-scan one. */
  title?: string;
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

export function cleanNoteText(rawText: string): string {
  if (!rawText) return "";
  let text = rawText.trim();

  // If text starts with '{' or contains '"synthesis":'
  if (text.startsWith("{") || text.includes('"synthesis":')) {
    try {
      const parsed = JSON.parse(text);
      if (typeof parsed.synthesis === "string") {
        text = parsed.synthesis;
      }
    } catch {
      // Try regex extracting synthesis string
      const match = text.match(/"synthesis"\s*:\s*"((?:\\.|[^"\\])*)/);
      if (match) {
        text = match[1].replace(/\\n/g, "\n").replace(/\\"/g, '"');
      } else {
        // Strip out {"synthesis": " and trailing ,"next_steps"...
        text = text
          .replace(/^\s*\{\s*"synthesis"\s*:\s*"?/, "")
          .replace(/"?\s*,?\s*"next_steps".*$/s, "")
          .replace(/[\{\}\[\]"]+/g, " ");
      }
    }
  }

  // Collapse 3 or more consecutive citations into at most 2
  text = text.replace(/(?:\[\d+\]\s*){3,}/g, match => {
    const citations = match.match(/\[\d+\]/g) || [];
    const unique = Array.from(new Set(citations));
    return " " + unique.slice(0, 2).join(" ");
  });

  return text.trim();
}

export function extractNextStepsIfMissing(
  synthesisText: string,
  currentSteps: string[]
): { cleanSynthesis: string; steps: string[] } {
  if (currentSteps && currentSteps.length > 0) {
    return {
      cleanSynthesis: cleanNoteText(synthesisText),
      steps: currentSteps.map(cleanNoteText).filter(Boolean),
    };
  }

  if (synthesisText.includes('"next_steps"')) {
    try {
      const parsed = JSON.parse(synthesisText);
      if (Array.isArray(parsed.next_steps) && parsed.next_steps.length > 0) {
        return {
          cleanSynthesis: cleanNoteText(parsed.synthesis || synthesisText),
          steps: parsed.next_steps.map((s: unknown) => cleanNoteText(String(s))).filter(Boolean),
        };
      }
    } catch {
      const match = synthesisText.match(/"next_steps"\s*:\s*\[(.*)/s);
      if (match) {
        const rawSteps = match[1].match(/"([^"\\]*(?:\\.[^"\\]*)*)(?:"|$)/g) || [];
        const extracted = rawSteps
          .map(s => s.replace(/^"/, "").replace(/"$/, "").trim())
          .map(cleanNoteText)
          .filter(Boolean);
        if (extracted.length > 0) {
          return {
            cleanSynthesis: cleanNoteText(synthesisText),
            steps: extracted,
          };
        }
      }
    }
  }

  return {
    cleanSynthesis: cleanNoteText(synthesisText),
    steps: (currentSteps || []).map(cleanNoteText).filter(Boolean),
  };
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

  // Filter out raw JSON fragments or empty sentences
  const validSentences = sentences
    .map(s => {
      let cleanText = s.text
        .replace(/^\s*\{\s*"synthesis"\s*:\s*"?/, "")
        .replace(/"?\s*,?\s*"next_steps"\s*:\s*\["?/, "")
        .replace(/["\}\]]+$/g, "")
        .trim();
      cleanText = cleanText.replace(/(?:\[\d+\]\s*){3,}/g, match => {
        const citations = match.match(/\[\d+\]/g) || [];
        const unique = Array.from(new Set(citations));
        return " " + unique.slice(0, 2).join(" ");
      }).trim();
      return { ...s, text: cleanText };
    })
    .filter(s => s.text.length > 5 && !s.text.startsWith('", "'));

  if (!validSentences.length) return null;

  return (
    <div className="space-y-1">
      <p className="text-[11px] font-medium text-muted-foreground">
        Per-claim support
      </p>
      {validSentences.map((sentence, idx) => {
        const number = passageNumber(sentence.supporting_passage_id);
        return (
          <div
            key={idx}
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
              {number === null ? "(no source)" : `[${number}]`}{" "}
              {stripCitationMarkers(sentence.text)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Removes the model's own `[n]` markers, which are never shown. */
function stripCitationMarkers(text: string): string {
  return text
    .replace(/\s*\[\d+\]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

/**
 * The source badge for one sentence: the passage the entailment model found
 * supporting it. Sentences the model did not judge supported get no badge, so
 * an unsupported claim reads as uncited rather than as backed by the nearest
 * passage.
 */
function EvidenceCitation({
  sentence,
  passages,
  onOpenEvidence,
}: {
  sentence: NoteSentence | undefined;
  passages: RetrievedPassage[];
  onOpenEvidence: () => void;
}) {
  if (sentence?.label !== "entailment" || !sentence.supporting_passage_id) {
    return null;
  }
  const index = passages.findIndex(
    passage => passage.passage_id === sentence.supporting_passage_id,
  );
  if (index === -1) return null;

  const passage = passages[index];
  const number = index + 1;
  const badgeClass =
    "inline-flex items-center text-[10px] font-semibold text-primary bg-primary/10 hover:bg-primary/25 hover:text-primary px-1 py-0.5 rounded mx-0.5 transition-colors cursor-pointer";

  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          {passage.url ? (
            <a
              href={passage.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={e => e.stopPropagation()}
              className={badgeClass}
              aria-label={`Supporting evidence [${number}]: ${passage.source}`}
            >
              [{number}]
            </a>
          ) : (
            <button
              type="button"
              onClick={onOpenEvidence}
              className={badgeClass}
              aria-label={`Supporting evidence [${number}]: ${passage.source}`}
            >
              [{number}]
            </button>
          )}
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs p-2 text-xs">
          <p className="font-medium text-foreground">
            [{number}] {passage.source}
          </p>
          <p className="text-[10px] text-muted-foreground mt-1 line-clamp-2">
            {passage.text}
          </p>
          {passage.url && (
            <span className="inline-flex items-center gap-1 text-[10px] text-primary mt-1.5 font-medium">
              Read original source on PMC <ExternalLink className="w-2.5 h-2.5" />
            </span>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * The synthesis, rendered sentence by sentence as the gate scored it, each with
 * its entailment-derived source. Falls back to the plain text, markers removed,
 * when no per-sentence verdicts came back.
 */
function SynthesisText({
  text,
  sentences,
  passages,
  onOpenEvidence,
}: {
  text: string;
  sentences: NoteSentence[];
  passages: RetrievedPassage[];
  onOpenEvidence: () => void;
}) {
  if (!sentences.length) return <span>{stripCitationMarkers(text)}</span>;
  return (
    <span>
      {sentences.map((sentence, idx) => (
        <span key={idx}>
          {idx > 0 && " "}
          {stripCitationMarkers(sentence.text)}
          <EvidenceCitation
            sentence={sentence}
            passages={passages}
            onOpenEvidence={onOpenEvidence}
          />
        </span>
      ))}
    </span>
  );
}

export function ClinicalNotePanel({
  title = "Clinical Interpretation",
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
  const band = response ? noteGroundingBand(response) : "unknown";
  const signedOff = status === "accepted" || status === "edited";

  const passageNumber = (passageId: string | null | undefined) => {
    if (!passageId || !response) return null;
    const index = response.retrieved_passages.findIndex(
      passage => passage.passage_id === passageId,
    );
    return index === -1 ? null : index + 1;
  };

  return (
    <section className="p-4 rounded-xl border bg-card" aria-label={title}>
      <div className="flex items-center justify-between gap-2 mb-3 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          <Stethoscope className="w-4 h-4 text-primary" aria-hidden="true" />
          <h3 className="text-sm font-medium">{title}</h3>
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

        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={onRegenerate}
                disabled={state.kind === "loading"}
                aria-label="Regenerate clinical note with MEDITRON"
                className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
              >
                <RefreshCw
                  className={cn("w-3.5 h-3.5", state.kind === "loading" && "animate-spin")}
                  aria-hidden="true"
                />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">
              <p className="text-xs">Regenerate clinical note with MEDITRON</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {state.kind === "loading" && (
        <div className="space-y-3 py-1" data-testid="clinical-note-loading" role="status">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" aria-hidden="true" />
            <span>Consulting clinical literature &amp; drafting note with MEDITRON…</span>
          </div>
          <div className="space-y-2">
            <Skeleton className="h-3.5 w-full" />
            <Skeleton className="h-3.5 w-[94%]" />
            <Skeleton className="h-3.5 w-[82%]" />
          </div>
          <div className="pt-1 space-y-1.5">
            <Skeleton className="h-3 w-[25%]" />
            <Skeleton className="h-3 w-[75%]" />
            <Skeleton className="h-3 w-[60%]" />
          </div>
        </div>
      )}

      {state.kind === "error" && (
        <div className="flex items-start justify-between gap-2 p-2 rounded-lg border bg-destructive/5 border-destructive/20">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-destructive flex-shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed text-destructive font-medium">
              {state.error.message || "Could not reach the clinical note service."}
            </p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={onRegenerate}
            className="text-xs h-7 flex-shrink-0"
          >
            <RefreshCw className="w-3 h-3 mr-1.5" aria-hidden="true" /> Try again
          </Button>
        </div>
      )}

      {/* Withheld: say so plainly, and say why. Never substitute other text. */}
      {state.kind === "ready" && !note && (
        <NoteAlert tone="warning" testId="clinical-note-unavailable">
          <div className="flex items-start justify-between gap-2">
            <div>
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
            </div>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onRegenerate}
              className="text-xs h-7 flex-shrink-0"
            >
              <RefreshCw className="w-3 h-3 mr-1.5" aria-hidden="true" /> Regenerate draft
            </Button>
          </div>
        </NoteAlert>
      )}

      {state.kind === "ready" && note && (() => {
        const { cleanSynthesis, steps } = extractNextStepsIfMissing(note.synthesis, note.next_steps);
        const passages = response?.retrieved_passages ?? [];
        const openEvidence = () => setShowEvidence(true);
        // The backend scores synthesis sentences first, then one entry per next
        // step, so the verdicts split at a fixed point.
        const stepCount = note.next_steps.length;
        const aligned = note.sentences.length >= stepCount;
        const synthesisSentences = aligned
          ? note.sentences.slice(0, note.sentences.length - stepCount)
          : [];
        const stepSentences = aligned
          ? note.sentences.slice(note.sentences.length - stepCount)
          : [];
        return (
          <div className="space-y-3" data-testid="clinical-note-body">
            <NoteAlert tone="warning">
              Drafted by a language model from retrieved literature. Requires
              physician review before it enters the patient record.
            </NoteAlert>

            {draft === null ? (
              <div className="p-3.5 rounded-lg border bg-card/60 text-xs sm:text-[13px] leading-relaxed text-foreground/90 shadow-2xs">
                <p className="font-normal leading-relaxed">
                  {reviewedText?.trim() ? (
                    reviewedText
                  ) : (
                    <SynthesisText
                      text={cleanSynthesis}
                      sentences={synthesisSentences}
                      passages={passages}
                      onOpenEvidence={openEvidence}
                    />
                  )}
                </p>
              </div>
            ) : (
              <Textarea
                value={draft}
                onChange={event => setDraft(event.target.value)}
                rows={6}
                aria-label="Edit clinical note"
                className="text-xs sm:text-sm leading-relaxed"
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
                  <SynthesisText
                    text={cleanSynthesis}
                    sentences={synthesisSentences}
                    passages={passages}
                    onOpenEvidence={openEvidence}
                  />
                </p>
              </details>
            )}

            {steps.length > 0 && (
              <div className="p-3 rounded-lg border bg-muted/20 space-y-2">
                <p className="text-[11px] font-semibold text-foreground/80 flex items-center gap-1.5 uppercase tracking-wider">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                  Recommended Next Steps
                </p>
                <ul className="space-y-1.5 pl-1">
                  {steps.map((step, idx) => (
                    <li
                      key={idx}
                      className="text-xs leading-relaxed flex items-start gap-2 text-foreground/90"
                    >
                      <span className="text-primary font-bold text-xs" aria-hidden="true">
                        {idx + 1}.
                      </span>
                      <span>
                        {stripCitationMarkers(step)}
                        <EvidenceCitation
                          sentence={stepSentences[idx]}
                          passages={passages}
                          onOpenEvidence={openEvidence}
                        />
                      </span>
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
                {response?.synthesis_groundedness != null ? (
                  <>
                    Synthesis {response.synthesis_groundedness.toFixed(3)} vs{" "}
                    {response.gate_threshold.toFixed(2)} · Next steps{" "}
                    {response.next_steps_groundedness != null
                      ? `${response.next_steps_groundedness.toFixed(3)} vs ${(
                          response.next_steps_threshold ?? response.gate_threshold
                        ).toFixed(2)}`
                      : "none"}{" "}
                    · overall {response.grounding_score?.toFixed(3) ?? "n/a"} ·{" "}
                  </>
                ) : (
                  <>
                    Groundedness {response?.grounding_score?.toFixed(3) ?? "n/a"} vs
                    threshold {response?.gate_threshold?.toFixed(2) ?? "n/a"} ·{" "}
                  </>
                )}
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
                  className="p-2.5 rounded-lg bg-muted/40 border text-[10px] leading-relaxed"
                >
                  <div className="flex items-center justify-between gap-2 mb-1 flex-wrap">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="font-semibold text-primary">[{index + 1}]</span>
                      <span className="text-foreground/90 font-medium">
                        {passage.source}
                      </span>
                      <span className="px-1 rounded bg-background text-muted-foreground uppercase text-[9px] font-medium border">
                        {passage.source_type}
                      </span>
                      <span className="px-1 rounded bg-background text-muted-foreground text-[9px] border">
                        {passage.licence}
                      </span>
                    </div>
                    {passage.url && (
                      <a
                        href={passage.url}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="inline-flex items-center gap-1 text-[10px] font-medium text-primary hover:underline px-1.5 py-0.5 rounded bg-primary/5 border border-primary/20 transition-colors"
                      >
                        <span>Read on PMC</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    )}
                  </div>
                  <p className="text-muted-foreground">{passage.text}</p>
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
        );
      })()}
    </section>
  );
}
