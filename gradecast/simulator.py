"""Monte Carlo simulation of the rest of the semester.

We "play out" the unfinished items thousands of times: each remaining item gets
a random score drawn from the predictor that suits it (Bayesian update for
quizzes and other items, gradient boosting for exams). Every run gives one
possible final total, and counting how often each letter grade appears gives
its probability. Everything is vectorised with numpy, so 10,000 runs are fast.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gradecast import bayes, model
from gradecast.course import Course, Item
from gradecast.grading import LETTERS, letters_for

# When a student has no quiz/other scores yet, assume a typical 75 coursework.
DEFAULT_COURSEWORK = 75.0


def _average(items: list[Item]) -> float | None:
    """Average score of already-taken items, or None if the list is empty."""
    if not items:
        return None
    return sum(item.score for item in items) / len(items)


def _draw_parameters(
    course: Course, exam_models: model.Models, item: Item
) -> tuple[float, float]:
    """Work out the (mean, sd) to draw random scores from for one remaining item.

    Args:
        course: The course, used to read the student's existing scores.
        exam_models: Trained final-exam models.
        item: A remaining (not yet scored) item.

    Returns:
        Mean and standard deviation of the score distribution.
    """
    quiz_scores = [i.score for i in course.taken("quiz")]
    other_scores = [i.score for i in course.taken("other")]

    if item.kind == "exam":
        # Coursework = average of all taken quiz and other scores.
        coursework = _average(course.taken("quiz") + course.taken("other"))
        if coursework is None:
            coursework = DEFAULT_COURSEWORK
        # If an exam was already taken, use it as the "midterm" input.
        taken_exams = course.taken("exam")
        midterm = taken_exams[0].score if taken_exams else None
        prediction = model.predict_exam(exam_models, coursework, midterm)
        return prediction.mean, prediction.sd

    if item.kind == "other":
        # Prefer the student's own "other" scores, then fall back to quizzes,
        # and finally to the plain prior (an empty list).
        evidence = other_scores or quiz_scores
    else:
        evidence = quiz_scores
    belief = bayes.update(evidence)
    return belief.mean, belief.sd


def simulate(
    course: Course,
    exam_models: model.Models,
    n: int = 10_000,
    pins: dict[str, float] | None = None,
    shift: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate ``n`` possible final course totals.

    Args:
        course: The course with whatever scores are known.
        exam_models: Trained final-exam models.
        n: Number of simulated semesters.
        pins: Item name -> score to fix in every run (overrides real scores).
        shift: Points added to every random draw ("study harder" knob).
        seed: Random seed, so the same inputs give the same answer.

    Returns:
        Array of ``n`` final totals (0-100).

    Example:
        >>> simulate(course, models, n=5).shape  # doctest: +SKIP
        (5,)
    """
    pins = pins or {}
    rng = np.random.default_rng(seed)

    # Points from fixed items (taken or pinned) are the same in every run.
    fixed_points = 0.0
    # Items still random, with their weights and score distributions.
    random_weights: list[float] = []
    random_means: list[float] = []
    random_sds: list[float] = []
    for item in course.items:
        if item.name in pins:
            fixed_points += pins[item.name] * item.weight / 100
        elif item.score is not None:
            fixed_points += item.score * item.weight / 100
        else:
            mean, sd = _draw_parameters(course, exam_models, item)
            random_weights.append(item.weight / 100)
            random_means.append(mean)
            random_sds.append(sd)

    # Nothing random left: every run gives the same total.
    if not random_weights:
        return np.full(n, fixed_points)

    # One n x k matrix of draws (rows = runs, columns = items) in one shot.
    # The shift moves every random mean, not the fixed scores.
    draws = rng.normal(
        loc=np.array(random_means) + shift,
        scale=np.array(random_sds),
        size=(n, len(random_weights)),
    )
    # Clip to 0-100 because a real test score can't be negative or above 100,
    # but random draws from a normal distribution sometimes go past those limits.
    draws = np.clip(draws, 0, 100)
    # Each run's total = fixed points + sum of (draw * weight) across items.
    return fixed_points + draws @ np.array(random_weights)


def grade_probabilities(totals: np.ndarray) -> dict[str, float]:
    """Share of simulated runs that ended in each letter grade.

    Args:
        totals: Simulated final totals.

    Returns:
        Letter -> probability, in A -> E order, summing to 1.
    """
    letters = letters_for(totals)
    return {letter: float(np.mean(letters == letter)) for letter in LETTERS}


@dataclass
class Scenario:
    """A named what-if preset.

    Attributes:
        name: Label shown to the user.
        build: Function turning a course into ``(pins, shift)``.
    """

    name: str
    build: Callable[[Course], tuple[dict[str, float], float]]


def _stable(course: Course) -> tuple[dict[str, float], float]:
    """No changes: the baseline prediction."""
    return {}, 0.0


def _bomb_final(course: Course) -> tuple[dict[str, float], float]:
    """Pin the Final Exam at 50, if it has not been taken yet."""
    still_open = {item.name for item in course.remaining()}
    return ({"Final Exam": 50.0} if "Final Exam" in still_open else {}), 0.0


def _skip_quiz(course: Course) -> tuple[dict[str, float], float]:
    """Pin the first remaining quiz at 0, if any quiz remains."""
    quizzes = course.remaining("quiz")
    return ({quizzes[0].name: 0.0} if quizzes else {}), 0.0


def _locked_in(course: Course) -> tuple[dict[str, float], float]:
    """Everything remaining goes 5 points better than expected."""
    return {}, 5.0


SCENARIOS: list[Scenario] = [
    Scenario("Stable", _stable),
    Scenario("Bomb the final", _bomb_final),
    Scenario("Skip a quiz", _skip_quiz),
    Scenario("Locked in", _locked_in),
]


def safe_zone(
    course: Course,
    exam_models: model.Models,
    item_name: str,
    target: float = 85,
    min_prob: float = 0.5,
    **sim_kwargs,
) -> float | None:
    """Lowest score on one item that keeps the chance of hitting ``target`` high.

    Uses binary search: the higher the pinned score, the higher the chance, so
    we repeatedly halve the range of possible scores.

    Args:
        course: The course.
        exam_models: Trained final-exam models.
        item_name: Item to search over, e.g. "Final Exam".
        target: Course total to reach, 85 = the A line.
        min_prob: Required probability of reaching the target.
        **sim_kwargs: Extra arguments for ``simulate`` (e.g. ``shift``, ``pins``).

    Returns:
        The score needed, 0.0 if already safe, None if even 100 is not enough.
    """

    def chance_at(score: float) -> float:
        """Probability of reaching target if the item is pinned at ``score``."""
        pins = {**sim_kwargs.get("pins", {}), item_name: score}
        options = {**sim_kwargs, "pins": pins}
        totals = simulate(course, exam_models, n=4000, **options)
        return float(np.mean(totals >= target))

    # Even a perfect score is not enough.
    if chance_at(100) < min_prob:
        return None
    # Even zero is enough, so nothing is needed.
    if chance_at(0) >= min_prob:
        return 0.0
    low, high = 0.0, 100.0
    for _ in range(12):
        middle = (low + high) / 2
        if chance_at(middle) >= min_prob:
            high = middle  # middle works, so try lower
        else:
            low = middle  # middle too low, so go higher
    return high
