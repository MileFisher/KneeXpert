"""The safety gate: decide whether a generated note may reach a clinician.

Pure. Takes *already-computed* entailment scores rather than calling the model
itself, which is a deliberate seam: thresholds, aggregation and retry policy are
the parts most worth testing and most likely to change, and keeping them free of
the 740 MB cross-encoder means they are testable in milliseconds.

The policy, from the project's own §2.9: every clinical statement must be
explicitly supported by the retrieved evidence. Sentences are scored
individually; the note's score is the mean of the per-sentence maxima, so one
fabricated claim among several supported ones still drags the note down rather
than being averaged into invisibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rag.config import GROUNDEDNESS_TAU, RETRY_TEMPERATURES, TOP_K, TOP_K_RETRY

EntailmentLabel = Literal["entailment", "neutral", "contradiction"]

SentenceClass = Literal["synthesis", "next_step"]
"""Which part of the note a sentence came from.

Tracked because the two behave differently against a corpus of research
literature: a synthesis sentence restates findings, while a recommendation
("refer for orthopaedic evaluation") is the kind of generic clinical advice that
research prose supports only weakly. Reporting them separately is what lets the
threshold policy be decided from data instead of guessed.
"""


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
    """The gate's verdict on one generated note."""

    passed: bool
    groundedness: float
    threshold: float
    sentence_scores: list[SentenceScore] = field(default_factory=list)

    @property
    def rejected_sentences(self) -> list[str]:
        """Sentences below the threshold, worst first.

        Fed back verbatim into the retry prompt: naming the exact sentence gives
        the model more to act on than a paraphrased complaint.
        """
        failing = [s for s in self.sentence_scores if s.groundedness < self.threshold]
        return [s.text for s in sorted(failing, key=lambda s: s.groundedness)]

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
    scores: list[SentenceScore], threshold: float = GROUNDEDNESS_TAU
) -> GateOutcome:
    """Decide whether a note passes.

    Gates on the aggregate only. The per-class figures are recorded but do not
    gate, pending the data needed to choose between one threshold and separate
    thresholds per sentence class.
    """
    groundedness = aggregate(scores)
    return GateOutcome(
        passed=bool(scores) and groundedness >= threshold,
        groundedness=groundedness,
        threshold=threshold,
        sentence_scores=list(scores),
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
