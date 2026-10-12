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
from rag.eval.run_ablation import CONFIGS, load_done, record_key
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


def test_both_modalities_are_present() -> None:
    modalities = {case["modality"] for case in build_payloads()}
    assert modalities == {"xray", "mri"}


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


def test_load_done_reads_prior_rows(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"case_id": "c1", "config": "rag_gate", "attempt": 0}) + "\n"
        + json.dumps({"case_id": "c1", "config": "rag_gate", "attempt": 1}) + "\n",
        encoding="utf-8",
    )
    done = load_done(path)
    assert record_key("c1", "rag_gate", 0) in done
    assert record_key("c1", "rag_gate", 2) not in done


def test_load_done_survives_a_truncated_final_line(tmp_path) -> None:
    """A killed spot instance leaves a partial write; resume must not refuse."""
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"case_id": "c1", "config": "rag_gate", "attempt": 0}) + "\n"
        + '{"case_id": "c2", "config": "rag_g',
        encoding="utf-8",
    )
    assert load_done(path) == {record_key("c1", "rag_gate", 0)}


def test_load_done_on_a_missing_file_is_empty(tmp_path) -> None:
    assert load_done(tmp_path / "absent.jsonl") == set()


# ── Threshold sweep ───────────────────────────────────────────────────────────


def rows_for(case_id: str, scores: list[float], config: str = "rag_gate") -> list[dict]:
    return [
        {
            "case_id": case_id,
            "config": config,
            "attempt": index,
            "groundedness": score,
        }
        for index, score in enumerate(scores)
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
