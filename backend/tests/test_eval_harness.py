"""Tests for the evaluation harness.

Pure: no models, no network. What matters here is not that the code runs but
that the *measurement* is sound — an ablation that silently loses coverage, or a
sweep that is only honest at one threshold, produces numbers that look fine and
mean nothing.
"""

from __future__ import annotations

import json

import pytest

from rag.eval.payloads import build_payloads, summarise
from rag.eval.run_ablation import CONFIGS, load_resumed, record_key
from rag.eval.sweep_tau import replay_config, sweep


# ── Payload set ───────────────────────────────────────────────────────────────


def test_payload_generation_is_deterministic() -> None:
    """A grader must be able to rebuild the committed set byte-for-byte."""
    first = json.dumps(build_payloads(), sort_keys=True)
    second = json.dumps(build_payloads(), sort_keys=True)
    assert first == second


def test_case_ids_are_unique() -> None:
    """They are the resume key, so a collision would silently skip work."""
    ids = [case["case_id"] for case in build_payloads()]
    assert len(ids) == len(set(ids))


def test_every_kl_grade_is_covered() -> None:
    grades = {case["grade"] for case in build_payloads()}
    assert grades == {0, 1, 2, 3, 4}


def test_all_sixteen_skm_tea_labels_are_covered() -> None:
    """Even coverage of rare pathologies is the reason the set is constructed."""
    import json as _json
    from pathlib import Path

    from rag.config import BACKBONE_ROOT

    catalogue = _json.loads(
        (BACKBONE_ROOT / "mri" / "categories.json").read_text(encoding="utf-8")
    )
    expected = {c["name"] for c in catalogue["categories"]}

    seen: set[str] = set()
    for case in build_payloads():
        for label in case["positive_labels"]:
            seen.add(label["name"])
    assert seen == expected, f"uncovered: {sorted(expected - seen)}"


def test_every_note_kind_is_present() -> None:
    modalities = {case["modality"] for case in build_payloads()}
    assert modalities == {"xray", "mri", "combined"}


def test_combined_cases_cover_agreement_and_disagreement_at_every_grade() -> None:
    combined = [c for c in build_payloads() if c["modality"] == "combined"]
    seen = {
        (c["studies"][0]["grade"], c["studies"][0]["grade"] == c["studies"][1]["grade"])
        for c in combined
    }
    assert seen == {(g, agree) for g in range(5) for agree in (True, False)}


def test_combined_cases_carry_one_xray_and_one_mri_study() -> None:
    for case in build_payloads():
        if case["modality"] == "combined":
            assert sorted(s["modality"] for s in case["studies"]) == ["mri", "xray"]


def test_combined_cases_record_the_orchestrator_grade() -> None:
    """The committed grade must be what the request validator will compute."""
    from rag.schemas import FeedbackGenerateRequest

    for case in build_payloads():
        if case["modality"] == "combined":
            req = FeedbackGenerateRequest.model_validate(case)
            assert (req.grade, req.confidence) == (case["grade"], case["confidence"])


def test_xray_cases_carry_no_labels() -> None:
    """Radiography cannot see soft tissue; sending labels would be incoherent."""
    for case in build_payloads():
        if case["modality"] == "xray":
            assert case["positive_labels"] == []


def test_confidence_spans_the_prompt_calibration_boundaries() -> None:
    """The prompt treats <70 as materially uncertain and >=85 as high."""
    confidences = {case["confidence"] for case in build_payloads()}
    assert any(c < 70 for c in confidences)
    assert any(70 <= c < 85 for c in confidences)
    assert any(c >= 85 for c in confidences)


def test_label_probabilities_are_percentages() -> None:
    for case in build_payloads():
        for label in case["positive_labels"]:
            assert 0 <= label["probability"] <= 100


def test_summary_reports_the_case_count() -> None:
    payloads = build_payloads()
    assert str(len(payloads)) in summarise(payloads)


# ── Resume bookkeeping ────────────────────────────────────────────────────────


def test_record_key_distinguishes_config_and_attempt() -> None:
    keys = {
        record_key("c1", config, attempt)
        for config in CONFIGS
        for attempt in range(3)
    }
    assert len(keys) == len(CONFIGS) * 3


def test_load_resumed_reads_prior_rows(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"case_id": "c1", "config": "rag_gate", "attempt": 0}) + "\n"
        + json.dumps({"case_id": "c1", "config": "rag_gate", "attempt": 1}) + "\n",
        encoding="utf-8",
    )
    resumed = load_resumed(path)
    assert record_key("c1", "rag_gate", 0) in resumed
    assert record_key("c1", "rag_gate", 2) not in resumed


def test_load_resumed_survives_a_truncated_final_line(tmp_path) -> None:
    """A killed spot instance leaves a partial write; resume must not refuse."""
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"case_id": "c1", "config": "rag_gate", "attempt": 0}) + "\n"
        + '{"case_id": "c2", "config": "rag_g',
        encoding="utf-8",
    )
    assert set(load_resumed(path)) == {record_key("c1", "rag_gate", 0)}


def test_load_resumed_on_a_missing_file_is_empty(tmp_path) -> None:
    assert load_resumed(tmp_path / "absent.jsonl") == {}


# ── Threshold sweep ───────────────────────────────────────────────────────────


def rows_for(
    case_id: str,
    scores: list[float],
    config: str = "rag_gate",
    next_steps: list[float | None] | None = None,
) -> list[dict]:
    """Stored attempts. ``scores`` are synthesis means; next steps optional."""
    steps = next_steps or [None] * len(scores)
    return [
        {
            "case_id": case_id,
            "config": config,
            "attempt": index,
            "groundedness": score,
            "synthesis_groundedness": score,
            "next_steps_groundedness": step,
        }
        for index, (score, step) in enumerate(zip(scores, steps, strict=True))
    ]


def test_serves_the_first_attempt_that_clears_the_threshold() -> None:
    rows = rows_for("c1", [0.40, 0.75, 0.90])
    result = replay_config(rows, "rag_gate", tau=0.70)
    assert result.served == 1
    assert result.mean_attempts == 2.0


def test_withholds_when_no_attempt_clears() -> None:
    result = replay_config(rows_for("c1", [0.10, 0.20, 0.30]), "rag_gate", tau=0.70)
    assert result.served == 0
    assert result.withheld == 1
    assert result.mean_attempts == 3.0


def test_a_score_exactly_at_the_threshold_is_served() -> None:
    assert replay_config(rows_for("c1", [0.70]), "rag_gate", tau=0.70).served == 1


# This is the property the whole offline-sweep design exists for: because every
# attempt was generated unconditionally, the replay can answer for thresholds
# *above* the one the run used, not only at or below it.
def test_sweep_is_valid_above_the_run_time_threshold() -> None:
    rows = rows_for("c1", [0.62, 0.81, 0.55])
    assert replay_config(rows, "rag_gate", tau=0.60).served == 1
    strict = replay_config(rows, "rag_gate", tau=0.80)
    assert strict.served == 1
    assert strict.mean_attempts == 2.0, "must find the later, stronger attempt"
    assert replay_config(rows, "rag_gate", tau=0.90).served == 0


def test_serve_rate_is_monotonically_non_increasing_in_tau() -> None:
    """A stricter threshold can never serve more notes."""
    rows = (
        rows_for("c1", [0.30, 0.55, 0.80])
        + rows_for("c2", [0.95, 0.20, 0.10])
        + rows_for("c3", [0.45, 0.48, 0.49])
    )
    series = sweep(rows, "rag_gate")
    rates = [row.serve_rate for row in series]
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))


def test_configurations_are_replayed_independently() -> None:
    rows = rows_for("c1", [0.90], "rag_gate") + rows_for("c1", [0.10], "ungrounded")
    assert replay_config(rows, "rag_gate", 0.70).served == 1
    assert replay_config(rows, "ungrounded", 0.70).served == 0


def test_rows_without_a_score_do_not_count_as_passing() -> None:
    """A failed generation has no groundedness; it must not be served."""
    rows = [
        {"case_id": "c1", "config": "rag_gate", "attempt": 0, "groundedness": None},
        {"case_id": "c1", "config": "rag_gate", "attempt": 1, "failure": "timeout"},
    ]
    assert replay_config(rows, "rag_gate", tau=0.0).served == 0


def test_sweep_covers_the_documented_range() -> None:
    series = sweep(rows_for("c1", [0.5]), "rag_gate")
    assert series[0].tau == pytest.approx(0.50)
    assert series[-1].tau == pytest.approx(0.95)


# ── Two thresholds (decision 17) ──────────────────────────────────────────────


def test_replay_requires_both_parts_to_clear() -> None:
    rows = rows_for("c1", [0.90], next_steps=[0.30])
    assert replay_config(rows, "rag_gate", tau=0.60).served == 0
    assert replay_config(rows, "rag_gate", tau=0.60, next_steps_tau=0.30).served == 1


def test_diagonal_sweep_moves_both_thresholds_together() -> None:
    series = sweep(rows_for("c1", [0.9], next_steps=[0.55]), "rag_gate")
    for row in series:
        assert row.next_steps_tau == row.tau
        assert row.served == (1 if row.tau <= 0.55 else 0)


def test_grid_zero_column_is_synthesis_only_gating() -> None:
    from rag.eval.sweep_tau import sweep_grid

    rows = rows_for("c1", [0.9], next_steps=[0.0])
    cells = {(r.tau, r.next_steps_tau): r for r in sweep_grid(rows, "rag_gate")}
    assert cells[(0.6, 0.0)].served == 1
    assert cells[(0.6, 0.05)].served == 0


def test_grid_serve_rate_never_rises_with_either_threshold() -> None:
    from rag.eval.sweep_tau import GRID_NEXT_STEPS_TAUS, GRID_SYNTHESIS_TAUS, sweep_grid

    rows = (
        rows_for("c1", [0.55, 0.80], next_steps=[0.9, 0.2])
        + rows_for("c2", [0.95, 0.70], next_steps=[0.4, 0.65])
        + rows_for("c3", [0.62, 0.61], next_steps=[None, 0.1])
    )
    rate = {(r.tau, r.next_steps_tau): r.serve_rate for r in sweep_grid(rows, "rag_gate")}
    for s, n in zip(GRID_SYNTHESIS_TAUS, GRID_SYNTHESIS_TAUS[1:]):
        for t in GRID_NEXT_STEPS_TAUS:
            assert rate[(n, t)] <= rate[(s, t)]
    for t, u in zip(GRID_NEXT_STEPS_TAUS, GRID_NEXT_STEPS_TAUS[1:]):
        for s in GRID_SYNTHESIS_TAUS:
            assert rate[(s, u)] <= rate[(s, t)]


def test_first_attempt_failures_are_attributed_to_each_part() -> None:
    """The question decision 17 was raised to answer: what does the gate fire on."""
    rows = (
        rows_for("synth-weak", [0.3], next_steps=[0.9])
        + rows_for("steps-weak", [0.9], next_steps=[0.3])
        + rows_for("both-weak", [0.3], next_steps=[0.3])
        + rows_for("no-steps", [0.9])
    )
    result = replay_config(rows, "rag_gate", tau=0.60)
    assert result.first_failed_synthesis == 2
    assert result.first_failed_next_steps == 2
    assert result.served == 1


def test_revision_hint_judges_each_class_against_its_recorded_threshold() -> None:
    from rag.eval.run_ablation import revision_hint_from_row

    row = {
        "run_time_tau": 0.6,
        "run_time_tau_next_steps": 0.4,
        "sentence_scores": [
            {"text": "Weak synthesis.", "sentence_class": "synthesis", "groundedness": 0.5},
            {"text": "Accepted step.", "sentence_class": "next_step", "groundedness": 0.5},
        ],
    }
    hint = revision_hint_from_row(row)
    assert hint is not None
    assert "Weak synthesis." in hint
    assert "Accepted step." not in hint


def test_rows_from_before_per_class_gating_reuse_the_single_threshold() -> None:
    from rag.eval.run_ablation import revision_hint_from_row

    row = {
        "run_time_tau": 0.6,
        "sentence_scores": [
            {"text": "Weak step.", "sentence_class": "next_step", "groundedness": 0.5},
        ],
    }
    hint = revision_hint_from_row(row)
    assert hint is not None and "Weak step." in hint


# ── The committed artefact must match the generator ──────────────────────────


def test_committed_payloads_json_matches_the_generator() -> None:
    """Reproducibility is only real if the committed file is pinned to the code.

    The earlier determinism test compared build_payloads() to itself, which
    passes even if payloads.json has drifted.
    """
    from rag.config import EVAL_PAYLOADS_PATH

    assert EVAL_PAYLOADS_PATH.is_file(), "payloads.json is not committed"
    on_disk = json.loads(EVAL_PAYLOADS_PATH.read_text(encoding="utf-8"))
    assert on_disk == build_payloads()


# ── The ablation loop itself ─────────────────────────────────────────────────


def _fake_attempt(monkeypatch, scores_by_attempt: dict[int, float]) -> list[dict]:
    """Replace generation and scoring, recording the prompt each attempt saw."""
    from rag.gate import GateOutcome, SentenceScore
    from rag.schemas import ClinicalNote, RetrievedPassage, TriadScores
    from rag.service import AttemptResult

    seen: list[dict] = []
    passage = RetrievedPassage(
        passage_id="kl-grading#04-00",
        source="S",
        text="t",
        score=1.0,
        licence="internal",
        source_type="internal",
    )

    def _run_attempt(req, passages, attempt, include_passages, revision_hint=None):
        seen.append({"attempt": attempt, "revision_hint": revision_hint})
        score = scores_by_attempt.get(attempt, 0.1)
        sentence = SentenceScore(
            text=f"Claim {attempt}.",
            sentence_class="synthesis",
            groundedness=score,
            label="entailment" if score >= 0.6 else "neutral",
            supporting_passage_id=passage.passage_id,
        )
        return AttemptResult(
            attempt=attempt,
            retrieval_k=6,
            note=ClinicalNote(synthesis=f"Claim {attempt}.", next_steps=[]),
            parse_mode="json",
            outcome=GateOutcome(
                passed=score >= 0.6,
                groundedness=score,
                threshold=0.6,
                sentence_scores=[sentence],
            ),
            triad=TriadScores(
                context_relevance=0.8 if include_passages else None,
                answer_relevance=0.7,
                groundedness=score,
            ),
            passages=[passage],
            generation_ms=10,
            gate_ms=20,
        )

    monkeypatch.setattr("rag.eval.run_ablation.run_attempt", _run_attempt)
    monkeypatch.setattr("rag.eval.run_ablation.to_schema", lambda ps: [passage])
    monkeypatch.setattr("rag.retriever.retrieve", lambda req, k=6: [passage])
    return seen


def _case() -> dict:
    return build_payloads()[0]


def test_every_attempt_is_generated_even_when_the_first_passes(monkeypatch) -> None:
    """The property the offline sweep depends on, pinned at the loop."""
    from rag.eval.run_ablation import run_case

    seen = _fake_attempt(monkeypatch, {0: 0.95, 1: 0.90, 2: 0.85})
    rows = list(run_case(_case(), "rag_gate", {}))  # type: ignore[arg-type]
    assert [r["attempt"] for r in rows] == [0, 1, 2]
    assert [s["attempt"] for s in seen] == [0, 1, 2]


def test_all_three_triad_dimensions_are_recorded(monkeypatch) -> None:
    """Without these the per-configuration triad table cannot be written."""
    from rag.eval.run_ablation import run_case

    _fake_attempt(monkeypatch, {0: 0.9})
    row = next(iter(run_case(_case(), "rag_gate", {})))  # type: ignore[arg-type]
    assert row["groundedness"] is not None
    assert row["answer_relevance"] is not None
    assert row["context_relevance"] is not None


def test_ungrounded_records_context_relevance_as_none(monkeypatch) -> None:
    """None, not zero: undefined where no context was supplied."""
    from rag.eval.run_ablation import run_case

    _fake_attempt(monkeypatch, {0: 0.9})
    row = next(iter(run_case(_case(), "ungrounded", {})))  # type: ignore[arg-type]
    assert row["context_relevance"] is None
    assert row["answer_relevance"] is not None


def test_a_failing_attempt_passes_its_rejections_to_the_next(monkeypatch) -> None:
    from rag.eval.run_ablation import run_case

    seen = _fake_attempt(monkeypatch, {0: 0.2, 1: 0.9})
    list(run_case(_case(), "rag_gate", {}))  # type: ignore[arg-type]
    assert seen[0]["revision_hint"] is None
    assert seen[1]["revision_hint"] is not None
    assert "Claim 0." in seen[1]["revision_hint"]


# The bug this test exists for: skipping a recorded attempt also skipped the
# revision update, so a resumed attempt ran with no REVISION block and the
# results file silently mixed two prompt conditions.
def test_resuming_still_carries_the_revision_hint_forward(monkeypatch) -> None:
    from rag.eval.run_ablation import record_key, run_case

    case = _case()
    seen = _fake_attempt(monkeypatch, {1: 0.9, 2: 0.9})
    resumed = {
        record_key(case["case_id"], "rag_gate", 0): {
            "case_id": case["case_id"],
            "config": "rag_gate",
            "attempt": 0,
            "run_time_tau": 0.6,
            "sentence_scores": [
                {"text": "Fabricated claim.", "groundedness": 0.05},
            ],
        }
    }
    list(run_case(case, "rag_gate", resumed))  # type: ignore[arg-type]

    assert [s["attempt"] for s in seen] == [1, 2], "recorded attempt re-generated"
    assert seen[0]["revision_hint"] is not None, "resumed run lost the revision hint"
    assert "Fabricated claim." in seen[0]["revision_hint"]
