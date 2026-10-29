"""Tests for the optional TruLens layer.

The first group runs everywhere and pins the fallback: with TruLens off or not
installed, the gate behaves exactly as without this module. The second group
needs TruLens installed (``requirements-trulens.txt``) and is skipped otherwise;
it records into a throwaway SQLite file, never the real trace log.
"""

from __future__ import annotations

import importlib.util

import pytest

from rag import trulens_layer
from rag.gate import SentenceScore, evaluate


def outcome_for(synthesis: float, steps: float | None, tau_s=0.6, tau_n=0.6):
    scores = [
        SentenceScore(text="S.", sentence_class="synthesis", groundedness=synthesis, label="entailment")
    ]
    if steps is not None:
        scores.append(
            SentenceScore(text="N.", sentence_class="next_step", groundedness=steps, label="entailment")
        )
    return evaluate(scores, threshold=tau_s, next_steps_threshold=tau_n)


@pytest.fixture
def fresh_tracer(monkeypatch):
    """Reset the process-wide singleton around each test."""
    monkeypatch.setattr(trulens_layer, "_tracer", None)
    monkeypatch.setattr(trulens_layer, "_tracer_error", None)
    yield
    monkeypatch.setattr(trulens_layer, "_tracer", None)


# ── Fallback: runs without TruLens ────────────────────────────────────────────


def test_off_by_default_the_gate_verdict_is_untouched(fresh_tracer, monkeypatch) -> None:
    monkeypatch.setattr(trulens_layer, "TRULENS_ENABLED", False)
    tracer = trulens_layer.get_tracer()
    assert tracer.active is False
    for synthesis, steps in [(0.9, 0.9), (0.9, 0.2), (0.3, 1.0)]:
        outcome = outcome_for(synthesis, steps)
        assert tracer.judge(outcome) is outcome


def test_enabled_but_unimportable_falls_back_and_says_why(fresh_tracer, monkeypatch) -> None:
    monkeypatch.setattr(trulens_layer, "TRULENS_ENABLED", True)

    def missing() -> None:
        raise ImportError("No module named 'trulens'")

    monkeypatch.setattr(trulens_layer, "_build_trulens_tracer", missing)
    assert trulens_layer.get_tracer().active is False
    status = trulens_layer.trulens_status()
    assert status["trulens_enabled"] is True
    assert status["trulens_active"] is False
    assert "No module named 'trulens'" in status["trulens_error"]


def test_the_null_tracer_runs_the_request_directly() -> None:
    tracer = trulens_layer._NullTracer()
    assert tracer.record("q", lambda: 42, str) == 42


# ── With TruLens installed ────────────────────────────────────────────────────

needs_trulens = pytest.mark.skipif(
    importlib.util.find_spec("trulens") is None,
    reason="TruLens not installed; pip install -r requirements-trulens.txt",
)


@pytest.fixture
def live_tracer(fresh_tracer, monkeypatch, tmp_path):
    monkeypatch.setattr(trulens_layer, "TRULENS_ENABLED", True)
    monkeypatch.setattr(trulens_layer, "TRULENS_DIR", tmp_path)
    monkeypatch.setattr(
        trulens_layer, "TRULENS_DB_URL", f"sqlite:///{(tmp_path / 't.sqlite').as_posix()}"
    )
    tracer = trulens_layer.get_tracer()
    assert tracer.active, trulens_layer.trulens_status()
    return tracer


def _span_events(tracer) -> list[dict]:
    tracer.session.force_flush()
    events = tracer.session.connector.get_events(app_name=trulens_layer.TRULENS_APP_NAME)
    return [a for a in events["record_attributes"] if isinstance(a, dict)]


@needs_trulens
def test_the_guardrail_reaches_the_same_verdict_as_the_gate(live_tracer) -> None:
    cases = [(0.9, 0.9), (0.9, 0.2), (0.3, 1.0), (0.6, 0.6), (0.9, None), (0.59, None)]
    for synthesis, steps in cases:
        outcome = outcome_for(synthesis, steps)
        assert live_tracer.judge(outcome).passed == outcome.passed, (synthesis, steps)


@needs_trulens
def test_a_lower_next_steps_threshold_is_honoured_by_the_guardrail(live_tracer) -> None:
    outcome = outcome_for(0.9, 0.45, tau_s=0.6, tau_n=0.4)
    assert outcome.passed
    assert live_tracer.judge(outcome).passed


@needs_trulens
def test_a_request_is_recorded_with_guardrail_and_triad_spans(live_tracer) -> None:
    from rag.schemas import TriadScores

    def run() -> str:
        live_tracer.retrieved("KL grade 3 knee", ["Definite joint space narrowing."])
        live_tracer.judge(outcome_for(0.3, 0.9))  # rejected attempt
        live_tracer.judge(outcome_for(0.9, 0.9))  # accepted attempt
        live_tracer.scored(
            TriadScores(context_relevance=0.81, answer_relevance=0.77, groundedness=0.9)
        )
        return "served"

    assert live_tracer.record("KL grade 3 knee", run, str) == "served"

    attrs = _span_events(live_tracer)
    span_types = {a.get("ai.observability.span_type") for a in attrs}
    assert {"record_root", "retrieval", "guardrail"} <= span_types
    verdicts = [
        a.get("ai.observability.guardrail.passed")
        for a in attrs
        if a.get("ai.observability.span_type") == "guardrail"
    ]
    assert sorted(verdicts) == [False, True]
    triad = [a for a in attrs if trulens_layer.ATTR_GROUNDEDNESS in a]
    assert triad and triad[0][trulens_layer.ATTR_GROUNDEDNESS] == pytest.approx(0.9)


@needs_trulens
def test_the_leaderboard_shows_the_rag_triad(live_tracer) -> None:
    from rag.schemas import TriadScores

    def run() -> str:
        live_tracer.scored(
            TriadScores(context_relevance=0.81, answer_relevance=0.77, groundedness=0.9)
        )
        return "served"

    live_tracer.record("KL grade 3 knee", run, str)
    live_tracer.session.force_flush()
    live_tracer.app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)
    live_tracer.session.force_flush()
    board = live_tracer.session.get_leaderboard()
    for metric, expected in [
        ("Groundedness", 0.9),
        ("Context Relevance", 0.81),
        ("Answer Relevance", 0.77),
    ]:
        assert metric in board.columns, list(board.columns)
        assert board[metric].iloc[0] == pytest.approx(expected)
