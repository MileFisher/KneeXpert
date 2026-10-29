"""Tests for grade fusion. Pure arithmetic, except the one endpoint test."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator import FusionRequest, StudyResult, fuse


def study(modality: str, grade: int, confidence: float) -> StudyResult:
    return StudyResult(modality=modality, grade=grade, confidence=confidence)  # type: ignore[arg-type]


def test_the_more_confident_study_pulls_the_grade_towards_it() -> None:
    # (2 x 90 + 4 x 60) / 150 = 2.8
    result = fuse([study("xray", 2, 90.0), study("mri", 4, 60.0)])
    assert result.weighted_mean == pytest.approx(2.8)
    assert result.grade == 3
    assert result.weights == {"xray": 0.6, "mri": 0.4}


def test_an_exact_half_rounds_up_not_to_even() -> None:
    """Python's round(2.5) is 2. A tie between grades must not depend on parity."""
    result = fuse([study("xray", 2, 80.0), study("mri", 3, 80.0)])
    assert result.weighted_mean == pytest.approx(2.5)
    assert result.grade == 3


def test_a_half_that_arrives_as_float_noise_still_rounds_up() -> None:
    # 33.3 x 2 + 33.3 x 3 over 66.6 is 2.5 exactly, but not in binary floating point.
    assert fuse([study("xray", 2, 33.3), study("mri", 3, 33.3)]).grade == 3


def test_confidence_is_the_plain_mean_never_boosted() -> None:
    """Two models agreeing is not evidence that either is more accurate. The old
    frontend added 4.2 points for agreement; that must not come back."""
    agreeing = fuse([study("xray", 3, 80.0), study("mri", 3, 90.0)])
    assert agreeing.confidence == 85.0


def test_agreement_and_gap() -> None:
    same = fuse([study("xray", 3, 80.0), study("mri", 3, 70.0)])
    assert (same.agreement, same.grade_gap) == ("concordant", 0)

    apart = fuse([study("xray", 1, 80.0), study("mri", 4, 70.0)])
    assert (apart.agreement, apart.grade_gap) == ("discordant", 3)


def test_zero_confidence_falls_back_to_equal_weights() -> None:
    result = fuse([study("xray", 1, 0.0), study("mri", 3, 0.0)])
    assert result.weighted_mean == pytest.approx(2.0)
    assert result.weights == {"xray": 0.5, "mri": 0.5}


def test_a_single_study_passes_through_unchanged() -> None:
    result = fuse([study("xray", 2, 77.7)])
    assert (result.grade, result.confidence, result.agreement) == (2, 77.7, "concordant")
    assert result.weights == {"xray": 1.0}


def test_nothing_to_fuse_is_an_error() -> None:
    with pytest.raises(ValueError):
        fuse([])


def test_a_request_may_not_repeat_a_modality() -> None:
    with pytest.raises(ValidationError):
        FusionRequest(studies=[study("xray", 1, 80.0), study("xray", 2, 80.0)])


def test_the_endpoint_returns_the_fused_result() -> None:
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)
    response = client.post(
        "/api/fusion",
        json={
            "studies": [
                {"modality": "xray", "grade": 2, "confidence": 90},
                {"modality": "mri", "grade": 4, "confidence": 60},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["grade"], body["agreement"], body["grade_gap"]) == (3, "discordant", 2)
    assert body["confidence"] == 75.0
    assert body["method"] == "confidence_weighted"


def test_the_endpoint_rejects_an_empty_request() -> None:
    from fastapi.testclient import TestClient

    import app as app_module

    response = TestClient(app_module.app).post("/api/fusion", json={"studies": []})
    assert response.status_code == 422
