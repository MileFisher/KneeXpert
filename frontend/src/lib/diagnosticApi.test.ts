import { afterEach, describe, expect, it, vi } from "vitest";

import { generateClinicalNote } from "./diagnosticApi";
import type { ClinicalNoteResponse } from "./diagnosticApi";

function jsonResponse(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: "OK",
    json: () => Promise.resolve(body),
  } as Response;
}

const WITHHELD: ClinicalNoteResponse = {
  clinical_note: null,
  generated_by: null,
  grounding_score: 0.21,
  triad: null,
  retrieved_passages: [
    {
      passage_id: "kl-grading#04-00",
      source: "KL grading — Grade 3",
      text: "Definite joint space narrowing, usually medial.",
      score: 1.0,
      licence: "internal",
      source_type: "internal",
    },
  ],
  gate_attempts: 3,
  gate_threshold: 0.6,
  fallback_reason: "gate_rejected",
  parse_mode: "json",
  latency_ms: { total: 24000 },
  config: "rag_gate",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("generateClinicalNote", () => {
  it("posts JSON to the feedback endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(WITHHELD));
    vi.stubGlobal("fetch", fetchMock);

    await generateClinicalNote({ modality: "xray", grade: 3, confidence: 81.4 });

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/api/feedback/generate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toMatchObject({
      modality: "xray",
      grade: 3,
      confidence: 81.4,
    });
  });

  it("returns a generated note on the happy path", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          ...WITHHELD,
          clinical_note: {
            synthesis: "Grade 3 change is present [1].",
            next_steps: ["Correlate with symptoms [1]."],
            sentences: [
              {
                text: "Grade 3 change is present [1].",
                groundedness: 0.96,
                label: "entailment",
                supporting_passage_id: "kl-grading#04-00",
                cited_marker: 1,
              },
            ],
          },
          generated_by: "meditron3-8b via Ollama",
          fallback_reason: null,
        }),
      ),
    );

    const result = await generateClinicalNote({
      modality: "xray",
      grade: 3,
      confidence: 81.4,
    });
    expect(result.clinical_note?.synthesis).toContain("Grade 3");
    expect(result.clinical_note?.sentences[0].label).toBe("entailment");
  });

  // A withheld note is a normal 200, not a failure. Treating it as an error
  // would make the UI unable to tell "not deployed" from "could not be grounded".
  it("parses a withheld note without throwing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(WITHHELD)));

    const result = await generateClinicalNote({
      modality: "mri",
      grade: 2,
      confidence: 58,
    });
    expect(result.clinical_note).toBeNull();
    expect(result.fallback_reason).toBe("gate_rejected");
    expect(result.retrieved_passages).toHaveLength(1);
  });

  it("throws the backend's detail message on an error status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { detail: "Retrieval index is empty. Run `python -m rag.ingest`." },
          { ok: false, status: 503 },
        ),
      ),
    );

    await expect(
      generateClinicalNote({ modality: "xray", grade: 1, confidence: 70 }),
    ).rejects.toThrow("Retrieval index is empty");
  });

  it("falls back to a generic message when the error body is unreadable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: () => Promise.reject(new Error("not json")),
      } as unknown as Response),
    );

    await expect(
      generateClinicalNote({ modality: "xray", grade: 1, confidence: 70 }),
    ).rejects.toThrow("Internal Server Error");
  });

  it("passes an abort signal through", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(WITHHELD));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await generateClinicalNote(
      { modality: "xray", grade: 3, confidence: 81 },
      controller.signal,
    );
    expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal);
  });
});
