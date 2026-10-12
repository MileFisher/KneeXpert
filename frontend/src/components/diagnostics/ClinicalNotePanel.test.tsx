import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ClinicalNotePanel } from "./ClinicalNotePanel";
import { XRAY_FINDINGS_BY_GRADE } from "@/lib/clinicalFeedback";
import type { ClinicalNoteResponse } from "@/lib/diagnosticApi";

const BASE: ClinicalNoteResponse = {
  clinical_note: null,
  generated_by: "meditron3-8b via Ollama",
  grounding_score: 0.698,
  triad: null,
  retrieved_passages: [
    {
      passage_id: "kl-grading#04-00",
      source: "KL grading — Grade 3",
      text: "Definite joint space narrowing, usually medial.",
      score: 1,
      licence: "internal",
      source_type: "internal",
    },
  ],
  gate_attempts: 1,
  gate_threshold: 0.6,
  fallback_reason: null,
  parse_mode: "json",
  latency_ms: { total: 11000 },
  config: "rag_gate",
};

const WITH_NOTE: ClinicalNoteResponse = {
  ...BASE,
  clinical_note: {
    synthesis: "Grade 3 change with medial narrowing [1].",
    next_steps: ["Correlate with reported symptoms [1]."],
    sentences: [],
  },
};

function renderPanel(props: Partial<Parameters<typeof ClinicalNotePanel>[0]> = {}) {
  const handlers = {
    onAccept: vi.fn(),
    onEdit: vi.fn(),
    onReject: vi.fn(),
    onRegenerate: vi.fn(),
  };
  render(
    <ClinicalNotePanel
      state="ready"
      response={WITH_NOTE}
      error={null}
      status={null}
      {...handlers}
      {...props}
    />,
  );
  return handlers;
}

describe("ClinicalNotePanel", () => {
  it("shows a skeleton while generating", () => {
    renderPanel({ state: "loading", response: null });
    expect(screen.getByTestId("clinical-note-loading")).toBeInTheDocument();
    expect(screen.queryByTestId("clinical-note-body")).not.toBeInTheDocument();
  });

  it("renders the note and marks it AI-generated", () => {
    renderPanel();
    expect(screen.getByText(/Grade 3 change with medial narrowing/)).toBeInTheDocument();
    expect(screen.getByText("AI-GENERATED")).toBeInTheDocument();
    expect(screen.getByText(/Correlate with reported symptoms/)).toBeInTheDocument();
  });

  it("shows a qualitative band, not a bare score, in the header", () => {
    // The threshold is provisional, so a raw figure in the header would read as
    // calibrated. It stays available behind the evidence disclosure.
    renderPanel();
    expect(screen.getByText("Evidence-supported")).toBeInTheDocument();
    expect(screen.queryByText(/0\.698/)).not.toBeInTheDocument();
  });

  it("reveals the raw groundedness figure on expanding the evidence", () => {
    renderPanel();
    fireEvent.click(screen.getByRole("button", { name: /Show evidence/ }));
    expect(screen.getByTestId("clinical-note-evidence")).toBeInTheDocument();
    expect(screen.getByText(/0\.698/)).toBeInTheDocument();
    expect(screen.getByText(/Definite joint space narrowing/)).toBeInTheDocument();
  });

  describe("when the note was withheld", () => {
    const rejected: ClinicalNoteResponse = {
      ...BASE,
      fallback_reason: "gate_rejected",
      gate_attempts: 3,
    };

    it("states that it is unavailable and why", () => {
      renderPanel({ response: rejected });
      expect(screen.getByTestId("clinical-note-unavailable")).toBeInTheDocument();
      expect(screen.getByText(/could not be grounded/i)).toBeInTheDocument();
      expect(screen.getByText(/3 attempts/)).toBeInTheDocument();
    });

    // The core honesty invariant. The deterministic FeedbackBundle is served
    // separately and always, so if its text ever appeared here a clinician could
    // not tell dictionary output from model reasoning.
    it("never substitutes deterministic feedback text", () => {
      renderPanel({ response: rejected });
      for (const finding of XRAY_FINDINGS_BY_GRADE[3] ?? []) {
        expect(screen.queryByText(finding)).not.toBeInTheDocument();
      }
      expect(screen.queryByTestId("clinical-note-body")).not.toBeInTheDocument();
    });

    it("offers no sign-off actions, since there is nothing to sign off", () => {
      renderPanel({ response: rejected });
      expect(screen.queryByRole("button", { name: /Accept/ })).not.toBeInTheDocument();
    });

    it("distinguishes an unreachable model from a failed gate", () => {
      renderPanel({ response: { ...BASE, fallback_reason: "llm_unavailable" } });
      expect(screen.getByText(/not reachable/i)).toBeInTheDocument();
    });
  });

  describe("sign-off", () => {
    it("accepts a note", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /Accept/ }));
      expect(handlers.onAccept).toHaveBeenCalledOnce();
    });

    it("rejects a note with a reason", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /Reject/ }));
      expect(handlers.onReject).toHaveBeenCalledWith(expect.any(String));
    });

    it("sends the edited text, not the original, when edited", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /Edit/ }));
      const box = screen.getByLabelText("Edit clinical note");
      fireEvent.change(box, { target: { value: "Clinician wording." } });
      fireEvent.click(screen.getByRole("button", { name: /Save & sign off/ }));
      expect(handlers.onEdit).toHaveBeenCalledWith("Clinician wording.");
    });

    it("hides the actions once a verdict is recorded", () => {
      renderPanel({ status: "accepted" });
      expect(screen.getByText("SIGNED OFF")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /Accept/ })).not.toBeInTheDocument();
    });

    it("shows the clinician's wording after an edit", () => {
      renderPanel({ status: "edited", reviewedText: "Revised by clinician." });
      expect(screen.getByText("Revised by clinician.")).toBeInTheDocument();
      expect(
        screen.queryByText(/Grade 3 change with medial narrowing/),
      ).not.toBeInTheDocument();
    });

    it("can request a regeneration", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /Regenerate/ }));
      expect(handlers.onRegenerate).toHaveBeenCalledOnce();
    });
  });

  it("surfaces a transport error", () => {
    renderPanel({
      state: "error",
      response: null,
      error: new Error("Retrieval index is empty."),
    });
    expect(screen.getByText(/Retrieval index is empty/)).toBeInTheDocument();
  });
});
