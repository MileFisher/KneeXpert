"""The safety gate: decide whether a generated note may reach a clinician.

Pure. Takes *already-computed* entailment scores rather than calling the model
itself, which is a deliberate seam: thresholds, aggregation and retry policy are
the parts most worth testing and most likely to change, and keeping them free of
the 740 MB cross-encoder means they are testable in milliseconds.

The policy, from the project's own §2.9: every clinical statement must be
explicitly supported by the retrieved evidence. Sentences are scored
individually, then averaged *within each part of the note*: the synthesis must
clear its threshold and the next steps must clear theirs (decision 17). One
fabricated claim still drags its part down rather than being averaged into
invisibility, and well-supported generic recommendations can no longer carry a
wrong diagnosis, which a single pooled mean allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

from rag.config import RETRY_TEMPERATURES, SYNTHESIS_TAU, TOP_K, TOP_K_RETRY

EntailmentLabel = Literal["entailment", "neutral", "contradiction"]

SentenceClass = Literal["synthesis", "next_step"]
"""Which part of the note a sentence came from.

Gated separately because the two behave differently against a corpus of
research literature: a synthesis sentence restates findings, while a
recommendation ("refer for orthopaedic evaluation") is the kind of generic
clinical advice that research prose may support only weakly. A threshold per
class is what lets the sweep measure that instead of guessing it.
"""


def passes(
    synthesis: float | None,
    next_steps: float | None,
    synthesis_threshold: float,
    next_steps_threshold: float,
) -> bool:
    """The gate's verdict from per-class means. The single statement of the rule.

    Shared by the live gate and the offline sweep, so a replayed verdict can never
    drift from a production one.

    A note with no synthesis fails: recommendations alone are not a note. A note
    with no next steps is judged on its synthesis, since a note that recommends
    nothing has made no recommendation to support.
    """
    if synthesis is None or synthesis < synthesis_threshold:
        return False
    return next_steps is None or next_steps >= next_steps_threshold


MARGIN_CAP = 10.0
"""Ceiling on :func:`margin`. A zero threshold would otherwise make it infinite,
which a trace attribute cannot carry."""


def margin(
    synthesis: float | None,
    next_steps: float | None,
    synthesis_threshold: float,
    next_steps_threshold: float,
) -> float:
    """The verdict of :func:`passes` as one number: at least 1.0 exactly when it passes.

    The weaker part's score as a fraction of its own threshold. This is what lets
    a single-score guardrail — TruLens's ``block_output``, thresholded at 1.0 —
    apply a two-threshold rule without a second copy of it.

    Division can land a ratio on 1.0 from just below, so the result is pinned to
    the side :func:`passes` chose; the two can never disagree.
    """
    if synthesis is None:
        return 0.0

    def ratio(score: float, threshold: float) -> float:
        return MARGIN_CAP if threshold <= 0 else min(score / threshold, MARGIN_CAP)

    value = ratio(synthesis, synthesis_threshold)
    if next_steps is not None:
        value = min(value, ratio(next_steps, next_steps_threshold))
    if passes(synthesis, next_steps, synthesis_threshold, next_steps_threshold):
        return max(value, 1.0)
    return min(value, math.nextafter(1.0, 0.0))


def rejected_texts(
    sentences: Iterable[tuple[str, SentenceClass, float]],
    synthesis_threshold: float,
    next_steps_threshold: float,
) -> list[str]:
    """Sentences below their own class's threshold, worst first.

    Takes plain tuples so a resumed evaluation run can rebuild the same list from
    stored rows that the live gate builds from :class:`SentenceScore`.
    """
    failing = [
        (score, text)
        for text, cls, score in sentences
        if score
        < (next_steps_threshold if cls == "next_step" else synthesis_threshold)
    ]
    return [text for _, text in sorted(failing, key=lambda item: item[0])]


@dataclass(frozen=True)
class SentenceScore:
    """One sentence's entailment verdict against the best-supporting passage."""

    text: str
    sentence_class: SentenceClass
    groundedness: float
    label: EntailmentLabel
    supporting_passage_id: str | None = None
    cited_marker: int | None = None

    @property
    def citation_agrees(self) -> bool | None:
        """Whether the model's own marker matches the entailment-derived source.

        None when the model cited nothing, so "did not cite" is not counted as
        "cited wrongly". Reported as a free side-metric: the surfaced citation is
        always the entailment-derived one, which cannot be fabricated.
        """
        if self.cited_marker is None or self.supporting_passage_id is None:
            return None
        return self.supporting_passage_id == f"[{self.cited_marker}]"


@dataclass(frozen=True)
class GateOutcome:
    """The gate's verdict on one generated note.

    ``groundedness`` is the pooled mean over every sentence. It no longer decides
    the verdict — the per-class means do — but it stays the note-level figure the
    RAG Triad reports.
    """

    passed: bool
    groundedness: float
    threshold: float
    sentence_scores: list[SentenceScore] = field(default_factory=list)
    next_steps_threshold: float | None = None
    """None means the same bar as ``threshold``."""

    @property
    def effective_next_steps_threshold(self) -> float:
        if self.next_steps_threshold is None:
            return self.threshold
        return self.next_steps_threshold

    @property
    def margin(self) -> float:
        """See :func:`margin`. At least 1.0 exactly when ``passed``."""
        return margin(
            self.synthesis_groundedness,
            self.next_steps_groundedness,
            self.threshold,
            self.effective_next_steps_threshold,
        )

    @property
    def rejected_sentences(self) -> list[str]:
        """Sentences below their class's threshold, worst first.

        Fed back verbatim into the retry prompt: naming the exact sentence gives
        the model more to act on than a paraphrased complaint.
        """
        return rejected_texts(
            ((s.text, s.sentence_class, s.groundedness) for s in self.sentence_scores),
            self.threshold,
            self.effective_next_steps_threshold,
        )

    @property
    def synthesis_groundedness(self) -> float | None:
        return _mean_of(
            [s for s in self.sentence_scores if s.sentence_class == "synthesis"]
        )

    @property
    def next_steps_groundedness(self) -> float | None:
        return _mean_of(
            [s for s in self.sentence_scores if s.sentence_class == "next_step"]
        )

    @property
    def citation_agreement_rate(self) -> float | None:
        """Share of cited sentences whose marker matched the entailed passage."""
        judged = [
            s.citation_agrees
            for s in self.sentence_scores
            if s.citation_agrees is not None
        ]
        if not judged:
            return None
        return sum(1 for agreed in judged if agreed) / len(judged)


def _mean_of(scores: list[SentenceScore]) -> float | None:
    if not scores:
        return None
    return sum(s.groundedness for s in scores) / len(scores)


def aggregate(scores: list[SentenceScore]) -> float:
    """Note-level groundedness: the mean of per-sentence maxima.

    Mean rather than minimum so a single hedging sentence does not veto an
    otherwise well-supported note, and rather than maximum so one supported
    sentence cannot carry several fabricated ones.

    An empty note scores 0.0: nothing to support is not the same as supported.
    """
    if not scores:
        return 0.0
    return sum(s.groundedness for s in scores) / len(scores)


def evaluate(
    scores: list[SentenceScore],
    threshold: float = SYNTHESIS_TAU,
    next_steps_threshold: float | None = None,
) -> GateOutcome:
    """Decide whether a note passes.

    The synthesis mean must clear ``threshold`` and the next-steps mean must
    clear ``next_steps_threshold``, which defaults to the same bar. Callers that
    serve notes pass both explicitly from config.
    """
    synthesis = _mean_of([s for s in scores if s.sentence_class == "synthesis"])
    next_steps = _mean_of([s for s in scores if s.sentence_class == "next_step"])
    steps_bar = threshold if next_steps_threshold is None else next_steps_threshold
    return GateOutcome(
        passed=passes(synthesis, next_steps, threshold, steps_bar),
        groundedness=aggregate(scores),
        threshold=threshold,
        sentence_scores=list(scores),
        next_steps_threshold=steps_bar,
    )


@dataclass(frozen=True)
class AttemptPlan:
    """Generation parameters for one attempt.

    Makes §2.9's "adjust generation parameters" a concrete, writable rule rather
    than a gesture.
    """

    attempt: int
    temperature: float
    retrieval_k: int
    include_revision_hint: bool


def plan_attempt(attempt: int, max_retries: int) -> AttemptPlan:
    """Escalation schedule: cool the sampling, then widen the evidence.

    Attempt 1 is the baseline. Attempt 2 keeps the same passages but names the
    rejected sentences, on the theory that the evidence was adequate and the
    writing was not. Attempt 3 assumes the opposite and retrieves more widely.
    """
    temperature = RETRY_TEMPERATURES[min(attempt, len(RETRY_TEMPERATURES) - 1)]
    return AttemptPlan(
        attempt=attempt,
        temperature=temperature,
        retrieval_k=TOP_K_RETRY if attempt >= max_retries else TOP_K,
        include_revision_hint=attempt > 0,
    )
