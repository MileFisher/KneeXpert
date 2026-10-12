"""Integration tests for POST /api/feedback/generate.

These run at the agreed seam: the real FastAPI app, real retrieval, real
embeddings and the real on-disk index — with **only the generation host faked**.
That is the one dependency worth faking: external, nondeterministic and metered.

Faking is done by monkeypatching `rag.ollama_client`, which is exactly what
pointing `OLLAMA_BASE_URL` at the dev stub does at runtime, minus the subprocess.

Requires the index to be built (`python -m rag.ingest`); skipped otherwise rather
than silently asserting against an empty corpus.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag.config import rag_index_ready

pytestmark = pytest.mark.skipif(
    not rag_index_ready(),
    reason="no retrieval index; run `python -m rag.ingest` first",
)

MRI_PAYLOAD = {
    "modality": "mri",
    "grade": 3,
    "confidence": 81.4,
    "positive_labels": [
        {"name": "Meniscal Tear (Radial)", "probability": 91.2},
        {"name": "Effusion", "probability": 84.7},
    ],
    "patient_context": {"age": 65, "gender": "Male", "bmi": 27.3, "pain_level": 6},
}

GROUNDED_JSON = json.dumps(
    {
        "synthesis": "Grade 3 change with medial narrowing [1].",
        "next_steps": ["Correlate with symptoms [1]."],
    }
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    import app as app_module

    return TestClient(app_module.app)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace only the generation host. Everything else stays real."""

    def _install(*, raw: str | None = None, exc: type[Exception] | None = None,
                 available: bool = True):
        from rag import ollama_client

        monkeypatch.setattr(ollama_client, "is_available", lambda *a, **k: available)

        def _chat(*args, **kwargs):
            if exc is not None:
                raise exc("stub")
            return raw if raw is not None else ""

        monkeypatch.setattr(ollama_client, "chat", _chat)

    return _install


# ── Happy path ────────────────────────────────────────────────────────────────


def test_returns_a_note_with_real_retrieved_passages(client, fake_llm) -> None:
    fake_llm(raw=GROUNDED_JSON)
    r = client.post("/api/feedback/generate", json=MRI_PAYLOAD)
    assert r.status_code == 200
    body = r.json()

    assert body["clinical_note"]["synthesis"].startswith("Grade 3 change")
    assert body["clinical_note"]["next_steps"] == ["Correlate with symptoms [1]."]
    assert body["generated_by"]
    assert body["fallback_reason"] is None
    assert body["retrieved_passages"], "retrieval produced nothing"
    assert body["gate_threshold"] > 0
    assert body["latency_ms"]["total"] >= 0

    passage = body["retrieved_passages"][0]
    assert passage["text"] and passage["licence"] and passage["source"]


def test_xray_payload_needs_no_labels(client, fake_llm) -> None:
    fake_llm(raw=GROUNDED_JSON)
    r = client.post(
        "/api/feedback/generate",
        json={"modality": "xray", "grade": 2, "confidence": 88.0},
    )
    assert r.status_code == 200
    assert r.json()["clinical_note"] is not None


# ── Degradation: 200 with a reason, never a 503, never substituted text ───────


def test_unreachable_host_yields_200_null_note_and_keeps_passages(
    client, fake_llm
) -> None:
    """The UI has to render this state, so it cannot be an error status."""
    fake_llm(available=False)
    r = client.post("/api/feedback/generate", json=MRI_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["clinical_note"] is None
    assert body["fallback_reason"] == "llm_unavailable"
    assert body["retrieved_passages"], "passages must survive a generation failure"


def test_timeout_is_reported_distinctly_from_unavailable(client, fake_llm) -> None:
    from rag.ollama_client import OllamaTimeout

    fake_llm(exc=OllamaTimeout)
    body = client.post("/api/feedback/generate", json=MRI_PAYLOAD).json()
    assert body["clinical_note"] is None
    assert body["fallback_reason"] == "timeout"


def test_empty_output_exhausts_retries_then_reports_parse_failure(
    client, fake_llm
) -> None:
    fake_llm(raw="")
    body = client.post("/api/feedback/generate", json=MRI_PAYLOAD).json()
    assert body["clinical_note"] is None
    assert body["fallback_reason"] == "parse_failed"
    assert body["gate_attempts"] >= 2, "should have retried before giving up"


def test_withheld_note_never_carries_deterministic_feedback_text(
    client, fake_llm
) -> None:
    """The honesty invariant: static text is never dressed as model output."""
    from feedback_engine import XRAY_FINDINGS

    fake_llm(available=False)
    body = client.post("/api/feedback/generate", json=MRI_PAYLOAD).json()
    serialised = json.dumps(body)
    for finding in XRAY_FINDINGS[3]:
        assert finding not in serialised
    assert body["generated_by"] is None


def test_malformed_prose_is_tolerated_rather_than_discarded(client, fake_llm) -> None:
    fake_llm(raw="The knee shows moderate degenerative change.")
    body = client.post("/api/feedback/generate", json=MRI_PAYLOAD).json()
    assert body["clinical_note"] is not None
    assert "moderate degenerative change" in body["clinical_note"]["synthesis"]


# ── Request validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        {"modality": "ct", "grade": 2, "confidence": 50},
        {"modality": "mri", "grade": 7, "confidence": 50},
        {"modality": "mri", "grade": 2, "confidence": 140},
        {"grade": 2, "confidence": 50},
    ],
)
def test_invalid_payloads_are_rejected(client, bad) -> None:
    assert client.post("/api/feedback/generate", json=bad).status_code == 422


# ── Configuration plumbing ────────────────────────────────────────────────────


def test_ungrounded_config_still_retrieves_passages(client, fake_llm) -> None:
    """They are scoring premises even when withheld from the prompt.

    Without this the ablation is ill-posed: groundedness would have nothing to
    score the ungrounded configuration against.
    """
    fake_llm(raw=GROUNDED_JSON)
    body = client.post(
        "/api/feedback/generate", json={**MRI_PAYLOAD, "config": "ungrounded"}
    ).json()
    assert body["config"] == "ungrounded"
    assert body["retrieved_passages"], "retrieval must run in every configuration"


def test_health_reports_the_layer_without_a_generation_host(client) -> None:
    body = client.get("/health").json()
    assert body["rag_index_ready"] is True
    assert body["rag_chunk_count"] > 0
    assert "llm_available" in body
