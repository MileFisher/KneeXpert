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
      state={{ kind: "ready", response: WITH_NOTE }}
      status={null}
      {...handlers}
      {...props}
    />,
  );
  return handlers;
}

describe("ClinicalNotePanel", () => {
  it("headed by its title, so a combined note is distinguishable", () => {
    renderPanel({ title: "Combined X-ray + MRI Note" });
    expect(
      screen.getByRole("heading", { name: "Combined X-ray + MRI Note" }),
    ).toBeInTheDocument();
  });

  it("shows a skeleton while generating", () => {
    renderPanel({ state: { kind: "loading" } });
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

  it("shows each part's score against its own threshold in the evidence", () => {
    renderPanel({
      state: {
        kind: "ready",
        response: {
          ...WITH_NOTE,
          synthesis_groundedness: 0.812,
          next_steps_groundedness: 0.455,
          next_steps_threshold: 0.4,
        },
      },
    });
    // Passed both bars, so supported — even though 0.455 < the 0.60 synthesis bar.
    expect(screen.getByText("Evidence-supported")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Show evidence/ }));
    const evidence = screen.getByTestId("clinical-note-evidence");
    expect(evidence.textContent).toMatch(/Synthesis 0\.812 vs 0\.60/);
    expect(evidence.textContent).toMatch(/Next steps 0\.455 vs 0\.40/);
  });

  describe("when the note was withheld", () => {
    const rejected: ClinicalNoteResponse = {
      ...BASE,
      fallback_reason: "gate_rejected",
      gate_attempts: 3,
    };

    it("states that it is unavailable and why", () => {
      renderPanel({ state: { kind: "ready", response: rejected } });
      expect(screen.getByTestId("clinical-note-unavailable")).toBeInTheDocument();
      expect(screen.getByText(/could not be grounded/i)).toBeInTheDocument();
      expect(screen.getByText(/3 attempts/)).toBeInTheDocument();
    });

    // The core honesty invariant. The deterministic FeedbackBundle is served
    // separately and always, so if its text ever appeared here a clinician could
    // not tell dictionary output from model reasoning.
    it("never substitutes deterministic feedback text", () => {
      renderPanel({ state: { kind: "ready", response: rejected } });
      for (const finding of XRAY_FINDINGS_BY_GRADE[3] ?? []) {
        expect(screen.queryByText(finding)).not.toBeInTheDocument();
      }
      expect(screen.queryByTestId("clinical-note-body")).not.toBeInTheDocument();
    });

    // A static reference list in the gap reads as the system's own evidence for
    // this case, and it bypasses the corpus licence gate.
    it.each(["gate_rejected", "llm_unavailable"] as const)(
      "shows no substitute reference list when %s",
      reason => {
        renderPanel({
          state: {
            kind: "ready",
            response: { ...BASE, fallback_reason: reason, gate_attempts: 3 },
          },
        });
        expect(screen.queryByText(/Verified Literature/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/Kellgren JH/)).not.toBeInTheDocument();
        expect(screen.queryAllByRole("link")).toHaveLength(0);
      },
    );

    it("offers no sign-off actions, since there is nothing to sign off", () => {
      renderPanel({ state: { kind: "ready", response: rejected } });
      expect(screen.queryByRole("button", { name: /Accept/ })).not.toBeInTheDocument();
    });

    it("distinguishes an unreachable model from a failed gate", () => {
      renderPanel({
        state: {
          kind: "ready",
          response: { ...BASE, fallback_reason: "llm_unavailable" },
        },
      });
      expect(screen.getByText(/not reachable/i)).toBeInTheDocument();
    });
  });

  describe("sign-off", () => {
    it("accepts a note", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /Accept/ }));
      expect(handlers.onAccept).toHaveBeenCalledOnce();
    });

    it("requires a reason before rejecting", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /^Reject/ }));
      const confirm = screen.getByRole("button", { name: /Confirm rejection/ });
      expect(confirm).toBeDisabled();
      expect(handlers.onReject).not.toHaveBeenCalled();
    });

    it("passes the clinician's own reason when rejecting", () => {
      const handlers = renderPanel();
      fireEvent.click(screen.getByRole("button", { name: /^Reject/ }));
      fireEvent.change(screen.getByLabelText("Reason for rejecting the note"), {
        target: { value: "Overstates the cartilage finding." },
      });
      fireEvent.click(screen.getByRole("button", { name: /Confirm rejection/ }));
      expect(handlers.onReject).toHaveBeenCalledWith(
        "Overstates the cartilage finding.",
      );
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

    it("shows the clinician's wording as the note body after an edit", () => {
      renderPanel({ status: "edited", reviewedText: "Revised by clinician." });
      expect(screen.getByText("Revised by clinician.")).toBeInTheDocument();
    });

    // "Reviewed by" is only meaningful if what was changed can be seen, so the
    // model's original wording stays available beside the approved version.
    it("keeps the model's original wording available after an edit", () => {
      renderPanel({ status: "edited", reviewedText: "Revised by clinician." });
      expect(
        screen.getByText("Show the model's original wording"),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Grade 3 change with medial narrowing/),
      ).toBeInTheDocument();
    });

    it("cleans raw JSON wrapper if model outputs unparsed JSON string", () => {
      const rawJsonNote: ClinicalNoteResponse = {
        ...BASE,
        clinical_note: {
          synthesis:
            '{"synthesis": "Patient exhibits joint-space narrowing [1] [2].", "next_steps": ["Order MRI [1]."]}',
          next_steps: [],
          sentences: [],
        },
      };
      renderPanel({ state: { kind: "ready", response: rawJsonNote } });
      expect(screen.getByText(/Patient exhibits joint-space narrowing/)).toBeInTheDocument();
      expect(screen.queryByText(/\{"synthesis"/)).not.toBeInTheDocument();
    });

    it("filters out hallucinated citations beyond available passages", () => {
      const runawayCitationNote: ClinicalNoteResponse = {
        ...BASE,
        clinical_note: {
          synthesis: "Finding documented [1] [191].",
          next_steps: ["Follow up [200]."],
          sentences: [],
        },
      };
      renderPanel({ state: { kind: "ready", response: runawayCitationNote } });
      expect(screen.getByText(/Finding documented/)).toBeInTheDocument();
      expect(screen.queryByText("[191]")).not.toBeInTheDocument();
      expect(screen.queryByText("[200]")).not.toBeInTheDocument();
    });

    it("can request a regeneration", () => {
      const handlers = renderPanel();
      const buttons = screen.getAllByRole("button", { name: /Regenerate/i });
      expect(buttons.length).toBeGreaterThan(0);
      fireEvent.click(buttons[0]);
      expect(handlers.onRegenerate).toHaveBeenCalledOnce();
    });
  });

  it("surfaces a transport error", () => {
    renderPanel({
      state: { kind: "error", error: new Error("Retrieval index is empty.") },
    });
    expect(screen.getByText(/Retrieval index is empty/)).toBeInTheDocument();
    expect(screen.queryByText(/Verified Literature/i)).not.toBeInTheDocument();
    expect(screen.queryAllByRole("link")).toHaveLength(0);
  });

  describe("citations", () => {
    const passages = [
      {
        passage_id: "kl-grading#04-00",
        source: "KL grading — Grade 3",
        text: "Definite joint space narrowing, usually medial.",
        score: 1,
        licence: "internal",
        source_type: "internal",
      },
      {
        passage_id: "PMC4772126#01-01",
        source: "Symptom–structure discordance",
        text: "Radiographic severity correlates weakly with pain.",
        score: 0.8,
        licence: "CC BY 4.0",
        source_type: "pmc",
        url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC4772126/",
      },
    ];

    // The model wrote [1] on both, but the entailment model found passage 2
    // supporting the step. Only the derived source may be shown.
    const cited: ClinicalNoteResponse = {
      ...BASE,
      retrieved_passages: passages,
      clinical_note: {
        synthesis: "Grade 3 change with medial narrowing [1]. Effusion is present [1].",
        next_steps: ["Correlate with reported symptoms [1]."],
        sentences: [
          {
            text: "Grade 3 change with medial narrowing [1].",
            groundedness: 0.91,
            label: "entailment",
            supporting_passage_id: "kl-grading#04-00",
            cited_marker: 1,
          },
          {
            text: "Effusion is present [1].",
            groundedness: 0.12,
            label: "neutral",
            supporting_passage_id: "kl-grading#04-00",
            cited_marker: 1,
          },
          {
            text: "Correlate with reported symptoms [1].",
            groundedness: 0.88,
            label: "entailment",
            supporting_passage_id: "PMC4772126#01-01",
            cited_marker: 1,
          },
        ],
      },
    };

    it("links each sentence to its entailed passage, not the model's marker", () => {
      renderPanel({ state: { kind: "ready", response: cited } });
      const link = screen.getByRole("link", { name: /Supporting evidence \[2\]/ });
      expect(link).toHaveAttribute(
        "href",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4772126/",
      );
      expect(
        screen.getByRole("button", { name: /Supporting evidence \[1\]/ }),
      ).toBeInTheDocument();
    });

    it("gives an unsupported sentence no citation", () => {
      renderPanel({ state: { kind: "ready", response: cited } });
      // Two supported sentences, two badges; the neutral one gets none.
      const badges = [
        ...screen.queryAllByRole("link", { name: /Supporting evidence/ }),
        ...screen.queryAllByRole("button", { name: /Supporting evidence/ }),
      ];
      expect(badges).toHaveLength(2);
    });

    it("never shows the model's own markers in the note text", () => {
      renderPanel({ state: { kind: "ready", response: WITH_NOTE } });
      const body = screen.getByTestId("clinical-note-body");
      expect(body.textContent).not.toMatch(/\[1\]/);
      expect(screen.getByText(/Grade 3 change with medial narrowing\./)).toBeInTheDocument();
    });
  });
});
