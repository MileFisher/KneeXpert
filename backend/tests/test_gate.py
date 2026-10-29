"""Tests for the safety gate's arithmetic and retry policy.

No models: the gate takes precomputed scores by design, so every case here runs
in microseconds on synthetic input.
"""

from __future__ import annotations

import pytest

from rag.gate import (
    SentenceScore,
    aggregate,
    evaluate,
    plan_attempt,
)


def score(
    value: float,
    *,
    text: str = "A claim.",
    cls: str = "synthesis",
    label: str = "entailment",
    passage: str | None = "[1]",
    marker: int | None = None,
) -> SentenceScore:
    return SentenceScore(
        text=text,
        sentence_class=cls,  # type: ignore[arg-type]
        groundedness=value,
        label=label,  # type: ignore[arg-type]
        supporting_passage_id=passage,
        cited_marker=marker,
    )


# ── Aggregation ───────────────────────────────────────────────────────────────


def test_aggregate_is_the_mean_of_sentence_scores() -> None:
    assert aggregate([score(0.9), score(0.5)]) == pytest.approx(0.7)


def test_empty_note_scores_zero_not_one() -> None:
    """Nothing to support is not the same as fully supported."""
    assert aggregate([]) == 0.0


def test_one_fabrication_drags_down_an_otherwise_supported_note() -> None:
    """Mean, not maximum: a supported sentence must not carry an unsupported one."""
    supported_only = aggregate([score(0.95), score(0.92), score(0.9)])
    with_fabrication = aggregate([score(0.95), score(0.92), score(0.05)])
    assert with_fabrication < supported_only
    assert with_fabrication < 0.7


def test_one_hedge_does_not_veto_a_strong_note() -> None:
    """Mean, not minimum: a single cautious sentence should not fail the note."""
    outcome = evaluate([score(0.95), score(0.93), score(0.40)], threshold=0.70)
    assert outcome.passed


# ── Threshold behaviour ───────────────────────────────────────────────────────


def test_score_exactly_at_the_threshold_passes() -> None:
    assert evaluate([score(0.70)], threshold=0.70).passed


def test_score_just_below_the_threshold_fails() -> None:
    assert not evaluate([score(0.6999)], threshold=0.70).passed


def test_empty_note_never_passes() -> None:
    assert not evaluate([], threshold=0.0).passed


def test_threshold_is_recorded_on_the_outcome() -> None:
    """Needed for the offline sweep: a stored run must say what it was judged at."""
    assert evaluate([score(0.8)], threshold=0.55).threshold == 0.55


# ── Rejected sentences feed the retry ─────────────────────────────────────────


def test_rejected_sentences_are_listed_worst_first() -> None:
    outcome = evaluate(
        [
            score(0.9, text="Supported."),
            score(0.1, text="Fabricated badly."),
            score(0.4, text="Weakly supported."),
        ],
        threshold=0.70,
    )
    assert outcome.rejected_sentences == ["Fabricated badly.", "Weakly supported."]


def test_passing_note_rejects_nothing() -> None:
    assert evaluate([score(0.9), score(0.85)], threshold=0.70).rejected_sentences == []


# ── Per-class reporting ───────────────────────────────────────────────────────


def test_synthesis_and_next_steps_are_scored_separately() -> None:
    """Recorded separately so the threshold policy can be chosen from data.

    Research prose supports a restatement of findings better than it supports a
    generic recommendation, and that difference has to be measurable.
    """
    outcome = evaluate(
        [
            score(0.9, cls="synthesis"),
            score(0.8, cls="synthesis"),
            score(0.3, cls="next_step"),
        ],
        threshold=0.70,
    )
    assert outcome.synthesis_groundedness == pytest.approx(0.85)
    assert outcome.next_steps_groundedness == pytest.approx(0.3)


def test_per_class_score_is_none_when_that_class_is_absent() -> None:
    outcome = evaluate([score(0.9, cls="synthesis")])
    assert outcome.next_steps_groundedness is None


# ── Per-class gating (decision 17) ────────────────────────────────────────────


def test_strong_next_steps_cannot_rescue_a_weak_synthesis() -> None:
    """The case that retired the pooled mean.

    Four weak synthesis sentences and three well-supported generic steps pool to
    0.63, which cleared a 0.60 bar — serving a wrong diagnosis on the strength of
    "correlate with symptoms".
    """
    note = [score(0.35, cls="synthesis")] * 4 + [score(1.0, cls="next_step")] * 3
    outcome = evaluate(note, threshold=0.60, next_steps_threshold=0.60)
    assert outcome.groundedness == pytest.approx(0.6286, abs=1e-4)
    assert not outcome.passed


def test_weak_next_steps_fail_a_note_with_a_strong_synthesis() -> None:
    outcome = evaluate(
        [
            score(0.95, cls="synthesis"),
            score(0.95, cls="synthesis"),
            score(0.2, cls="next_step"),
        ],
        threshold=0.70,
    )
    assert outcome.next_steps_groundedness == pytest.approx(0.2)
    assert not outcome.passed


def test_a_lower_next_steps_threshold_admits_weaker_recommendations() -> None:
    note = [score(0.9, cls="synthesis"), score(0.45, cls="next_step")]
    assert not evaluate(note, threshold=0.60, next_steps_threshold=0.60).passed
    assert evaluate(note, threshold=0.60, next_steps_threshold=0.40).passed


def test_zero_next_steps_threshold_is_synthesis_only_gating() -> None:
    """Option (a) is a corner of the same rule, not a separate code path."""
    note = [score(0.9, cls="synthesis"), score(0.0, cls="next_step")]
    assert evaluate(note, threshold=0.60, next_steps_threshold=0.0).passed


def test_next_steps_threshold_defaults_to_the_synthesis_threshold() -> None:
    outcome = evaluate([score(0.9, cls="synthesis")], threshold=0.65)
    assert outcome.next_steps_threshold == 0.65
    assert outcome.effective_next_steps_threshold == 0.65


def test_both_thresholds_are_recorded_on_the_outcome() -> None:
    outcome = evaluate(
        [score(0.9, cls="synthesis")], threshold=0.65, next_steps_threshold=0.5
    )
    assert (outcome.threshold, outcome.next_steps_threshold) == (0.65, 0.5)


def test_a_note_without_next_steps_is_judged_on_its_synthesis() -> None:
    assert evaluate([score(0.8, cls="synthesis")], threshold=0.60).passed


def test_recommendations_without_a_synthesis_never_pass() -> None:
    assert not evaluate([score(1.0, cls="next_step")], threshold=0.0).passed


def test_each_class_is_exactly_at_its_threshold_passes() -> None:
    note = [score(0.60, cls="synthesis"), score(0.40, cls="next_step")]
    assert evaluate(note, threshold=0.60, next_steps_threshold=0.40).passed


def test_rejected_sentences_are_judged_against_their_own_class() -> None:
    """The retry hint must name what failed *its* bar, or attempt 2 is asked to
    fix a recommendation the gate had accepted."""
    outcome = evaluate(
        [
            score(0.5, text="Weak synthesis.", cls="synthesis"),
            score(0.5, text="Acceptable step.", cls="next_step"),
            score(0.2, text="Weak step.", cls="next_step"),
        ],
        threshold=0.60,
        next_steps_threshold=0.40,
    )
    assert outcome.rejected_sentences == ["Weak step.", "Weak synthesis."]


def test_passes_is_the_rule_the_sweep_replays() -> None:
    from rag.gate import passes

    assert passes(0.7, None, 0.6, 0.6)
    assert passes(0.7, 0.5, 0.6, 0.5)
    assert not passes(0.7, 0.49, 0.6, 0.5)
    assert not passes(0.59, 1.0, 0.6, 0.0)
    assert not passes(None, 1.0, 0.0, 0.0)


# ── Citation agreement ────────────────────────────────────────────────────────


def test_matching_marker_counts_as_agreement() -> None:
    outcome = evaluate([score(0.9, passage="[1]", marker=1)])
    assert outcome.citation_agreement_rate == 1.0


def test_mismatched_marker_counts_as_disagreement() -> None:
    outcome = evaluate([score(0.9, passage="[2]", marker=1)])
    assert outcome.citation_agreement_rate == 0.0


def test_uncited_sentences_are_excluded_not_counted_as_wrong() -> None:
    """"Did not cite" is a different failure from "cited the wrong source"."""
    outcome = evaluate([score(0.9, marker=None), score(0.9, passage="[1]", marker=1)])
    assert outcome.citation_agreement_rate == 1.0


def test_agreement_is_none_when_nothing_was_cited() -> None:
    assert evaluate([score(0.9, marker=None)]).citation_agreement_rate is None


# ── Retry escalation ──────────────────────────────────────────────────────────


def test_attempt_escalation_cools_sampling_then_widens_evidence() -> None:
    from rag.config import TOP_K, TOP_K_RETRY

    first = plan_attempt(0, max_retries=2)
    second = plan_attempt(1, max_retries=2)
    third = plan_attempt(2, max_retries=2)

    assert (first.temperature, second.temperature, third.temperature) == (0.2, 0.1, 0.0)
    assert first.retrieval_k == TOP_K
    assert second.retrieval_k == TOP_K
    assert third.retrieval_k == TOP_K_RETRY


def test_first_attempt_carries_no_revision_hint() -> None:
    assert plan_attempt(0, max_retries=2).include_revision_hint is False
    assert plan_attempt(1, max_retries=2).include_revision_hint is True


def test_attempt_beyond_the_schedule_does_not_crash() -> None:
    plan = plan_attempt(9, max_retries=2)
    assert plan.temperature == 0.0
