"""A stand-in for the generation host, for development without a GPU.

    python -m uvicorn rag.dev.fake_ollama:app --port 11434
    FAKE_OLLAMA_MODE=fabricated python -m uvicorn rag.dev.fake_ollama:app --port 11434

Because ``OLLAMA_BASE_URL`` already defaults to ``localhost:11434``, pointing the
backend at this stub requires **no application code change at all** — which is the
point: the switch from stub to real hardware cannot leave an untested path behind.

Three canned responses, selected by ``FAKE_OLLAMA_MODE`` or a ``?mode=`` query:

``grounded``     a note whose claims follow from the supplied passages
``fabricated``   a confident note asserting things no passage supports, for
                 checking that the safety gate actually rejects something
``malformed``    prose instead of JSON, for the tolerant-parse path

Not shipped in the Docker image and never used in production.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, Request

app = FastAPI(title="fake-ollama", version="0.1.0")

MODEL_NAME = os.getenv("FAKE_OLLAMA_MODEL", "meditron3-8b")
DEFAULT_MODE = os.getenv("FAKE_OLLAMA_MODE", "grounded")

GROUNDED = {
    "synthesis": (
        "Radiographic findings are consistent with Kellgren-Lawrence grade 3 "
        "osteoarthritis, with definite joint space narrowing and osteophyte "
        "formation at multiple margins [1]. Subchondral sclerosis in the medial "
        "tibial plateau supports this grading [2]. Model confidence is moderate, "
        "so clinical correlation is advised before acting on the grade alone [1]."
    ),
    "next_steps": [
        "Correlate the radiographic grade with reported symptoms, which are known "
        "to diverge from structural findings [1].",
        "Consider MRI if soft-tissue pathology is clinically suspected, since "
        "cartilage and meniscal injury are not visible on plain radiography [2].",
    ],
}

# Deliberately unsupported: specific measurements, a named implant, a timeframe,
# and a surgical recommendation that no retrieved passage licenses. The gate
# should reject this.
FABRICATED = {
    "synthesis": (
        "MRI demonstrates a 4.2 cm full-thickness osteochondral defect of the "
        "lateral femoral condyle with complete ACL avulsion [1]. The patient "
        "requires immediate total knee arthroplasty within 72 hours to prevent "
        "irreversible collapse [2]. Serum CRP of 48 mg/L confirms septic "
        "arthritis [1]."
    ),
    "next_steps": [
        "Admit for emergency arthroplasty using a cemented posterior-stabilised "
        "implant [1].",
        "Start intravenous vancomycin 1 g twice daily for six weeks [2].",
    ],
}

MALFORMED = (
    "Sure! Here's my assessment of the knee. The imaging suggests moderate "
    "degenerative change, though I would want to see prior films before "
    "committing to a grade."
)


def _body_for(mode: str) -> str:
    if mode == "fabricated":
        return json.dumps(FABRICATED)
    if mode == "malformed":
        return MALFORMED
    return json.dumps(GROUNDED)


@app.get("/api/tags")
async def tags() -> dict[str, object]:
    """Mimic the model listing the real host exposes, so probes succeed."""
    return {
        "models": [
            {
                "name": MODEL_NAME,
                "model": MODEL_NAME,
                "size": 4_900_000_000,
                "details": {"family": "llama", "parameter_size": "8B"},
            }
        ]
    }


@app.post("/api/chat")
async def chat(request: Request) -> dict[str, object]:
    payload = await request.json()
    mode = request.query_params.get("mode", DEFAULT_MODE)

    # Surface what the backend actually sent, so a developer can see at a glance
    # whether num_ctx was set and how large the prompt really was.
    options = payload.get("options") or {}
    prompt_chars = sum(len(m.get("content", "")) for m in payload.get("messages", []))
    print(
        f"[fake-ollama] mode={mode} num_ctx={options.get('num_ctx')} "
        f"temp={options.get('temperature')} seed={options.get('seed')} "
        f"prompt_chars={prompt_chars} format={'yes' if payload.get('format') else 'no'}"
    )
    if options.get("num_ctx") in (None, 2048):
        print(
            "[fake-ollama] WARNING: num_ctx is unset or 2048. The real host would "
            "silently truncate the prompt from the left, dropping the system "
            "message's safety rules first."
        )

    return {
        "model": payload.get("model", MODEL_NAME),
        "done": True,
        "message": {"role": "assistant", "content": _body_for(mode)},
    }
