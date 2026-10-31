"""Tests for GradeCast's core logic (grading, course maths, bayes, model, simulator)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from gradecast import bayes, model
from gradecast.course import DEMO_SCORES, PRESETS, Course, Item
from gradecast.grading import letters_for, to_letter
from gradecast.simulator import SCENARIOS, grade_probabilities, safe_zone, simulate

# Where the committed, trained model lives.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gb_quantile.joblib"


def test_to_letter_boundaries() -> None:
    # Lower bounds are inclusive, so exactly 85 is an A but 84.99 is an A-.
    assert to_letter(85) == "A"
    assert to_letter(84.99) == "A-"
    assert to_letter(55) == "C"
    assert to_letter(39.9) == "E"


def test_letters_for_matches_to_letter() -> None:
    scores = np.array([100.0, 85.0, 84.99, 62.0, 40.0, 10.0])
    assert letters_for(scores).tolist() == [to_letter(s) for s in scores]


def test_deterministic_course_math() -> None:
    course = Course(
        "Example",
        [
            Item("A", 5, "other", 90),
            Item("B", 5, "quiz", 80),
            Item("C", 30, "exam", 75),
            Item("D", 5, "quiz"),
            Item("E", 5, "quiz"),
            Item("F", 50, "exam"),
        ],
    )
    assert course.current_total() == 31
    assert course.remaining_weight() == 60
    assert course.required_average() == 90
    assert course.best_case() == 91


def test_preset_weights_sum_to_100() -> None:
    assert PRESETS["Statistics & Probability"].total_weight() == 100


def test_bayes_more_scores_means_less_uncertainty() -> None:
    few = bayes.update([80.0, 82.0])
    many = bayes.update([80.0, 82.0] * 5)
    assert many.sd < few.sd


def test_bayes_no_scores_returns_prior_mean() -> None:
    assert bayes.update([]).mean == 75.0


# ---- model and simulator tests (use the trained model committed in models/) ----


@pytest.fixture(scope="module")
def exam_models() -> model.Models:
    """Load the trained exam models once for all tests in this file."""
    return model.load(MODEL_PATH)


def _demo_course() -> Course:
    """A fresh copy of the preset with the example scores filled in."""
    course = copy.deepcopy(PRESETS["Statistics & Probability"])
    for item in course.items:
        item.score = DEMO_SCORES.get(item.name)
    return course


def test_predict_exam_is_ordered_with_minimum_sd(exam_models: model.Models) -> None:
    for midterm in (None, 40.0, 90.0):
        prediction = model.predict_exam(exam_models, 75.0, midterm)
        assert prediction.low <= prediction.mean <= prediction.high
        assert prediction.sd >= 4


def test_all_pinned_gives_identical_totals(exam_models: model.Models) -> None:
    course = _demo_course()
    pins = {item.name: 90.0 for item in course.remaining()}
    totals = simulate(course, exam_models, n=500, pins=pins)
    assert np.all(totals == totals[0])
    probabilities = grade_probabilities(totals)
    assert max(probabilities.values()) == 1.0


def test_grade_probabilities_sum_to_one(exam_models: model.Models) -> None:
    totals = simulate(_demo_course(), exam_models)
    assert abs(sum(grade_probabilities(totals).values()) - 1) < 1e-9


def test_same_seed_same_output_and_shift_helps(exam_models: model.Models) -> None:
    course = _demo_course()
    first = simulate(course, exam_models, seed=7)
    second = simulate(course, exam_models, seed=7)
    assert np.array_equal(first, second)
    boosted = simulate(course, exam_models, seed=7, shift=10)
    assert boosted.mean() > first.mean()


def test_bombing_final_lowers_chance_of_a(exam_models: model.Models) -> None:
    # A strong student, so an A is actually reachable and the drop is visible.
    course = _demo_course()
    strong_scores = {"Class & Tutorial Participation": 95, "Quiz 1": 95, "Quiz 2": 92,
                     "Quiz 3": 94, "Programming Exercise": 95, "Mid Term Exam": 90}
    for item in course.items:
        item.score = strong_scores.get(item.name)
    stable = grade_probabilities(simulate(course, exam_models))["A"]
    pins, shift = SCENARIOS[1].build(course)
    bombed = grade_probabilities(simulate(course, exam_models, pins=pins, shift=shift))["A"]
    assert 0 < stable <= 1
    assert bombed < stable


def test_safe_zone_is_in_range_or_none(exam_models: model.Models) -> None:
    result = safe_zone(_demo_course(), exam_models, "Final Exam")
    assert result is None or 0 <= result <= 100
