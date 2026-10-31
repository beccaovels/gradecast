"""Tests for the read-only explanation helpers and the simulate refactor."""

import copy
from pathlib import Path

import numpy as np
import pytest

from gradecast import model
from gradecast.course import DEMO_SCORES, PRESETS, Course
from gradecast.explain import (
    a_status,
    belief_curves,
    convergence_snapshots,
    exam_forecasts,
    format_probability,
    plain_summary,
    sample_paths,
)
from gradecast.simulator import grade_probabilities, simulate, simulate_items

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gb_quantile.joblib"


@pytest.fixture(scope="module")
def exam_models() -> model.Models:
    """Load the trained exam models once."""
    return model.load(MODEL_PATH)


def _course(scores: dict[str, float]) -> Course:
    """The preset with the given scores filled in."""
    course = copy.deepcopy(PRESETS["Statistics & Probability"])
    for item in course.items:
        item.score = scores.get(item.name)
    return course


def test_simulate_output_unchanged_from_v1(exam_models: model.Models) -> None:
    # Reference values recorded from the original simulate() before the refactor.
    totals = simulate(_course(DEMO_SCORES), exam_models)
    assert np.isclose(totals.sum(), 773744.5383406151, rtol=1e-12)
    assert np.allclose(totals[:3], [78.50293276782531, 75.91735419743245, 78.30567354631577])
    pinned = simulate(
        _course(DEMO_SCORES), exam_models, pins={"Final Exam": 50.0, "Quiz 4": 0.0}, shift=3.0, seed=7
    )
    assert np.isclose(pinned.sum(), 660892.2433233301, rtol=1e-12)


def test_simulate_items_matches_simulate(exam_models: model.Models) -> None:
    course = _course(DEMO_SCORES)
    names, draws = simulate_items(course, exam_models, n=200)
    assert names == [item.name for item in course.remaining()]
    assert draws.shape == (200, len(names))
    weights = np.array([item.weight / 100 for item in course.remaining()])
    fixed = sum(item.score * item.weight / 100 for item in course.taken())
    assert np.allclose(fixed + draws @ weights, simulate(course, exam_models, n=200))


def test_simulate_items_nothing_random(exam_models: model.Models) -> None:
    course = _course({item.name: 80.0 for item in PRESETS["Statistics & Probability"].items})
    names, draws = simulate_items(course, exam_models, n=10)
    assert names == [] and draws.shape == (10, 0)


def test_sample_paths_shape_and_determinism(exam_models: model.Models) -> None:
    course = _course(DEMO_SCORES)
    first = sample_paths(course, exam_models, k=25)
    second = sample_paths(course, exam_models, k=25)
    assert first["cumulative"].shape == (25, len(course.items) + 1)
    assert np.array_equal(first["cumulative"], second["cumulative"])
    assert np.all(first["cumulative"][:, 0] == 0)
    assert np.all(np.diff(first["cumulative"], axis=1) >= 0)
    # Scored items are the same in every sampled semester.
    assert first["fixed"][0] and not first["fixed"][-1]
    assert len(first["grades"]) == 25 and np.isfinite(first["finals"]).all()


def test_convergence_snapshots_are_prefixes() -> None:
    totals = np.arange(10_000, dtype=float)
    snapshots = convergence_snapshots(totals)
    assert [len(s) for s in snapshots] == [10, 100, 1000, 10000]
    assert np.array_equal(snapshots[0], totals[:10])


def test_belief_curves_narrow_with_more_scores() -> None:
    scores = [88.0, 90.0, 86.0, 89.0]
    none = belief_curves(scores, 0)
    some = belief_curves(scores, 4)
    assert np.allclose(none["prior"], none["posterior"])
    assert some["posterior_belief"].sd < none["posterior_belief"].sd
    assert some["posterior_belief"].mean > none["posterior_belief"].mean
    assert some["posterior"].max() > none["posterior"].max()


def test_exam_forecasts(exam_models: model.Models) -> None:
    forecasts = exam_forecasts(_course(DEMO_SCORES), exam_models)
    assert [f["name"] for f in forecasts] == ["Final Exam"]
    forecast = forecasts[0]
    assert forecast["low"] <= forecast["median"] <= forecast["high"]
    assert forecast["midterm"] == 72 and forecast["sd"] >= 4
    assert np.isfinite(forecast["needed_for_a"])


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.0, "<1%"), (0.004, "<1%"), (0.005, "1%"), (0.372, "37%"), (0.994, "99%"), (0.995, ">99%"), (1.0, ">99%")],
)
def test_format_probability(p: float, expected: str) -> None:
    assert format_probability(p) == expected


def test_plain_summary_three_cases(exam_models: model.Models) -> None:
    normal = _course(DEMO_SCORES)
    probabilities = grade_probabilities(simulate(normal, exam_models))
    assert a_status(normal) == "open"
    assert "average 94" in plain_summary(normal, probabilities)
    assert "B+" in plain_summary(normal, probabilities)

    secured = _course({item.name: 100.0 for item in normal.items if item.name != "Quiz 6"})
    assert a_status(secured) == "secured"
    assert "already" in plain_summary(secured, probabilities)

    impossible = _course({"Mid Term Exam": 10.0, "Quiz 1": 0.0, "Quiz 2": 0.0})
    assert a_status(impossible) == "impossible"
    assert "out of reach" in plain_summary(impossible, probabilities)
