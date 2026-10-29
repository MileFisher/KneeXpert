"""TruLens integration for the clinical-note gate. Optional.

With ``RAG_TRULENS=1`` and TruLens installed:

* **The gate's pass/fail is decided by a TruLens guardrail.** ``block_output``
  thresholds :func:`rag.gate.margin` at 1.0, which is at least 1.0 exactly when
  both parts of the note clear their own thresholds. Each verdict is recorded as
  a GUARDRAIL span carrying its score, threshold and outcome.
* **Every note request is a TruLens record**: the retrieval, each attempt's
  verdict, and the RAG Triad scores the pipeline computed. Three TruLens metrics
  read those scores back, so the dashboard's leaderboard shows the RAG Triad.

Otherwise every hook is a no-op and the gate decides exactly as before.

Nothing here scores anything. The NLI model and embedder in ``rag.triad``
compute every number; TruLens decides on them and records them. That keeps one
implementation of each metric, and the dashboard shows the figures the gate
actually used rather than a second, possibly different, computation.

TruLens is imported lazily, inside :func:`_build_trulens_tracer`, so the backend
starts without it. It pins ``rich<14``, so it belongs in the backend's own
virtualenv or the Docker image (``requirements-trulens.txt``).
"""

from __future__ import annotations

import contextvars
import dataclasses
import importlib.util
import logging
import threading
from typing import Any, Callable, Protocol, TypeVar

from rag.config import (
    NEXT_STEPS_TAU,
    OLLAMA_MODEL,
    SYNTHESIS_TAU,
    TRULENS_APP_NAME,
    TRULENS_DB_URL,
    TRULENS_DIR,
    TRULENS_ENABLED,
)
from rag.gate import GateOutcome
from rag.schemas import TriadScores

logger = logging.getLogger(__name__)

T = TypeVar("T")

GATE_METRIC_NAME = "Clinical note gate (two-part NLI)"
TRIAD_SPAN = "kneexpert.rag_triad"
ATTR_GROUNDEDNESS = "kneexpert.rag_triad.groundedness"
ATTR_CONTEXT_RELEVANCE = "kneexpert.rag_triad.context_relevance"
ATTR_ANSWER_RELEVANCE = "kneexpert.rag_triad.answer_relevance"


class NoteTracer(Protocol):
    """What the note pipeline calls. Two implementations: no-op and TruLens."""

    active: bool

    def record(self, query: str, run: Callable[[], T], output_of: Callable[[T], str]) -> T:
        """Run one note request as a single TruLens record."""
        ...

    def retrieved(self, query: str, contexts: list[str]) -> None: ...

    def judge(self, outcome: GateOutcome) -> GateOutcome:
        """Return the outcome with ``passed`` set by the gate's decider."""
        ...

    def scored(self, triad: TriadScores | None) -> None: ...


class _NullTracer:
    """TruLens off or absent: the gate's own verdict stands, nothing is recorded."""

    active = False

    def record(self, query: str, run: Callable[[], T], output_of: Callable[[T], str]) -> T:
        return run()

    def retrieved(self, query: str, contexts: list[str]) -> None:
        return None

    def judge(self, outcome: GateOutcome) -> GateOutcome:
        return outcome

    def scored(self, triad: TriadScores | None) -> None:
        return None


class _Verdict(str):
    """The guarded value: reads as the margin, carries the outcome it came from.

    ``block_output`` passes the decorated function's return value to its metric,
    and a ``str`` survives TruLens's handling of it.
    """

    outcome: GateOutcome


def _verdict_margin(verdict: str) -> float:
    """The guardrail's metric: the gate's margin, already computed by the NLI."""
    return float(verdict.outcome.margin)  # type: ignore[attr-defined]


def _attribute_value(score: float) -> float:
    """Metric implementation that reports a score the pipeline already computed."""
    return float(score)


# The callable a record runs, passed through a context variable rather than as an
# argument so TruLens never tries to serialise it, and so concurrent requests in
# FastAPI's threadpool cannot see each other's.
_RUN: contextvars.ContextVar[Callable[[], str]] = contextvars.ContextVar("kneexpert_run")


def _build_trulens_tracer() -> NoteTracer:
    """Create the TruLens session, app and metrics. Imports TruLens."""
    from trulens.apps.app import TruApp
    from trulens.core import Metric, TruSession
    from trulens.core.database.connector.default import DefaultDBConnector
    from trulens.core.feedback.selector import Selector
    from trulens.core.guardrails.base import block_output
    from trulens.core.otel.instrument import instrument
    from trulens.otel.semconv.trace import SpanAttributes as SA

    TRULENS_DIR.mkdir(parents=True, exist_ok=True)
    session = TruSession(connector=DefaultDBConnector(database_url=TRULENS_DB_URL))

    gate_metric = Metric(implementation=_verdict_margin, name=GATE_METRIC_NAME)

    class _NotePipeline:
        """The instrumented surface TruLens records. Each method is a span."""

        @instrument()
        def respond(self, query: str) -> str:
            return _RUN.get()()

        @instrument(
            span_type=SA.SpanType.RETRIEVAL,
            attributes={
                SA.RETRIEVAL.QUERY_TEXT: "query",
                SA.RETRIEVAL.RETRIEVED_CONTEXTS: "contexts",
            },
        )
        def retrieved(self, query: str, contexts: list[str]) -> list[str]:
            return contexts

        @block_output(feedback=gate_metric, threshold=1.0, return_value=None)
        def judge(self, verdict: str) -> str:
            return verdict

        @instrument(
            name=TRIAD_SPAN,
            attributes={
                ATTR_GROUNDEDNESS: "groundedness",
                ATTR_CONTEXT_RELEVANCE: "context_relevance",
                ATTR_ANSWER_RELEVANCE: "answer_relevance",
            },
        )
        def scored(
            self,
            groundedness: float,
            context_relevance: float | None,
            answer_relevance: float,
        ) -> None:
            return None

    def triad_metric(name: str, attribute: str) -> Any:
        return Metric(implementation=_attribute_value, name=name).on(
            {
                "score": Selector(
                    span_name=TRIAD_SPAN,
                    span_attribute=attribute,
                    collect_list=False,
                    ignore_none_values=True,
                )
            }
        )

    pipeline = _NotePipeline()
    app = TruApp(
        pipeline,
        app_name=TRULENS_APP_NAME,
        # Thresholds and model in the version, so the leaderboard separates runs
        # made under different gate settings instead of averaging them together.
        app_version=f"{OLLAMA_MODEL} tau {SYNTHESIS_TAU:.2f}/{NEXT_STEPS_TAU:.2f}",
        main_method=pipeline.respond,
        feedbacks=[
            triad_metric("Groundedness", ATTR_GROUNDEDNESS),
            triad_metric("Context Relevance", ATTR_CONTEXT_RELEVANCE),
            triad_metric("Answer Relevance", ATTR_ANSWER_RELEVANCE),
        ],
    )
    return _TruLensTracer(session, app, pipeline)


class _TruLensTracer:
    """TruLens on: the guardrail decides, and each request is recorded."""

    active = True

    def __init__(self, session: Any, app: Any, pipeline: Any) -> None:
        self.session = session
        self.app = app
        self.pipeline = pipeline

    def record(self, query: str, run: Callable[[], T], output_of: Callable[[T], str]) -> T:
        holder: dict[str, T] = {}

        def run_and_render() -> str:
            holder["result"] = run()
            return output_of(holder["result"])

        token = _RUN.set(run_and_render)
        try:
            with self.app:
                self.pipeline.respond(query)
        finally:
            _RUN.reset(token)
        return holder["result"]

    def retrieved(self, query: str, contexts: list[str]) -> None:
        self.pipeline.retrieved(query, contexts)

    def judge(self, outcome: GateOutcome) -> GateOutcome:
        verdict = _Verdict(f"{outcome.margin:.6f}")
        verdict.outcome = outcome
        passed = self.pipeline.judge(verdict) is not None
        if passed != outcome.passed:
            # Unreachable by construction (rag.gate.margin is pinned to passes);
            # logged rather than asserted so a TruLens change cannot take the
            # endpoint down.
            logger.error(
                "TruLens guardrail disagreed with the gate: %s vs %s (margin %s)",
                passed,
                outcome.passed,
                outcome.margin,
            )
        return dataclasses.replace(outcome, passed=passed)

    def scored(self, triad: TriadScores | None) -> None:
        if triad is None:
            return
        self.pipeline.scored(
            triad.groundedness, triad.context_relevance, triad.answer_relevance
        )


_tracer: NoteTracer | None = None
_tracer_error: str | None = None
_lock = threading.Lock()


def get_tracer() -> NoteTracer:
    """The process-wide tracer. Lazy: TruLens is imported on first use, if at all.

    Falls back to the no-op tracer, with the reason logged and reported on
    ``/health``, when TruLens is enabled but cannot be imported.
    """
    global _tracer, _tracer_error
    if _tracer is None:
        with _lock:
            if _tracer is None:
                if not TRULENS_ENABLED:
                    _tracer = _NullTracer()
                else:
                    try:
                        _tracer = _build_trulens_tracer()
                    except ImportError as e:
                        _tracer_error = f"{type(e).__name__}: {e}"
                        logger.warning(
                            "RAG_TRULENS=1 but TruLens is unavailable (%s); "
                            "the gate runs without it.",
                            _tracer_error,
                        )
                        _tracer = _NullTracer()
    return _tracer


def trulens_status() -> dict[str, Any]:
    """The TruLens slice of ``/health``. Cheap: never imports TruLens itself."""
    return {
        "trulens_enabled": TRULENS_ENABLED,
        "trulens_installed": importlib.util.find_spec("trulens") is not None,
        "trulens_active": bool(_tracer is not None and _tracer.active),
        "trulens_error": _tracer_error,
    }
