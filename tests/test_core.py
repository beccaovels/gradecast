"""Tests for GradeCast's core logic (grading, course maths, bayes, model, simulator)."""

import numpy as np

from gradecast import bayes
from gradecast.course import PRESETS, Course, Item
from gradecast.grading import letters_for, to_letter


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
