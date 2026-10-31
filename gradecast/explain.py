"""Read-only helpers that let the app show how the model reached its answer.

Nothing here changes a prediction. These functions repackage what the
simulator and predictors already compute (sample semesters, belief curves,
exam forecasts, plain-English summaries) into shapes that are easy to draw.
There is no Streamlit code, so everything can be tested on its own.
"""

import numpy as np

from gradecast import bayes, model
from gradecast.course import Course
from gradecast.grading import letters_for, to_letter
from gradecast.simulator import DEFAULT_COURSEWORK, _draw_parameters, simulate_items

# The course total that earns an A.
A_LINE = 85.0


def format_probability(p: float) -> str:
    """Format a probability without ever showing a misleading 0% or 100%.

    Args:
        p: Probability between 0 and 1.

    Returns:
        "<1%" below 0.5%, ">99%" from 99.5% up, otherwise a rounded percent.

    Example:
        >>> format_probability(0.372)
        '37%'
    """
    # A tiny but non-zero chance is not the same as "impossible".
    if p < 0.005:
        return "<1%"
    # Likewise "almost certain" is not "guaranteed".
    if p >= 0.995:
        return ">99%"
    # Add 0.5 and truncate so 12.5 rounds up (Python's round() rounds to even).
    return f"{int(p * 100 + 0.5)}%"


def a_status(course: Course, target: float = A_LINE) -> str:
    """Say whether the A is out of reach, already secured, or still open.

    Args:
        course: The course.
        target: Course total that earns an A.

    Returns:
        "secured", "impossible" or "open".
    """
    # Already enough points banked: nothing left can take them away.
    if course.current_total() >= target:
        return "secured"
    # Even perfect scores on everything left would fall short.
    if course.best_case() < target:
        return "impossible"
    return "open"


def most_likely_letter(probabilities: dict[str, float]) -> str:
    """Letter grade with the highest probability."""
    return max(probabilities, key=probabilities.get)


def plain_summary(
    course: Course, probabilities: dict[str, float], target: float = A_LINE
) -> str:
    """One friendly sentence describing where the student stands.

    Args:
        course: The course.
        probabilities: Letter -> probability from the simulation.
        target: Course total that earns an A.

    Returns:
        A sentence for the hero banner (three cases: impossible, secured, normal).
    """
    status = a_status(course, target)
    if status == "secured":
        return "You have already banked enough points for an A, whatever happens next."
    if status == "impossible":
        best = course.best_case()
        return (
            f"An A is out of reach: even perfect scores on everything left would give "
            f"{best:.1f}. The highest grade you can still reach is {to_letter(best)}."
        )
    needed = course.required_average(target)
    likely = most_likely_letter(probabilities)
    # "an A" and "an E", but "a B+" or "a C".
    article = "an" if likely[0] in "AE" else "a"
    return (
        f"To get an A you need to average {needed:.0f} on what's left. "
        f"Right now you're most likely heading for {article} {likely}."
    )


def belief_curves(
    scores: list[float], n_used: int, prior_mean: float = 75.0, prior_sd: float = 12.0
) -> dict:
    """Bell curves for 'a typical student' versus 'you after N scores'.

    Args:
        scores: The student's scores, oldest first.
        n_used: How many of the first scores to use (0 = none).
        prior_mean: Typical score before seeing any data.
        prior_sd: Spread of true levels around that typical score.

    Returns:
        Dict with the x grid, the ``prior`` and ``posterior`` densities, and
        the matching ``prior_belief`` / ``posterior_belief`` objects.
    """
    # Both curves describe one FUTURE score, as in the simulator.
    prior_belief = bayes.update([], prior_mean, prior_sd)
    posterior_belief = bayes.update(scores[:n_used], prior_mean, prior_sd)
    x = np.linspace(0, 100, 201)

    def density(belief: bayes.Belief) -> np.ndarray:
        """Normal probability density of a belief over the x grid."""
        z = (x - belief.mean) / belief.sd
        return np.exp(-0.5 * z**2) / (belief.sd * np.sqrt(2 * np.pi))

    return {
        "x": x,
        "prior": density(prior_belief),
        "posterior": density(posterior_belief),
        "prior_belief": prior_belief,
        "posterior_belief": posterior_belief,
    }


def expected_scores(
    course: Course, exam_models: model.Models
) -> dict[str, tuple[float, float]]:
    """Expected score and spread of every remaining item, as the simulator sees them.

    Args:
        course: The course.
        exam_models: Trained final-exam models.

    Returns:
        Item name -> (mean, sd).
    """
    return {
        item.name: _draw_parameters(course, exam_models, item)
        for item in course.remaining()
    }


def exam_forecasts(
    course: Course, exam_models: model.Models, target: float = A_LINE
) -> list[dict]:
    """Forecast for every remaining exam, plus the score needed for an A.

    Args:
        course: The course.
        exam_models: Trained final-exam models.
        target: Course total that earns an A.

    Returns:
        One dict per remaining exam with name, low, median, high, sd, the
        inputs used (coursework average and midterm or None), and
        ``needed_for_a``: the score on this exam that reaches the target if
        every other remaining item scores as expected.
    """
    quiz_and_other = course.taken("quiz") + course.taken("other")
    coursework = (
        sum(item.score for item in quiz_and_other) / len(quiz_and_other)
        if quiz_and_other
        else DEFAULT_COURSEWORK
    )
    taken_exams = course.taken("exam")
    midterm = taken_exams[0].score if taken_exams else None
    expected = expected_scores(course, exam_models)

    forecasts: list[dict] = []
    for item in course.remaining("exam"):
        prediction = model.predict_exam(exam_models, coursework, midterm)
        # Points the OTHER remaining items are expected to add.
        others = sum(
            expected[other.name][0] * other.weight / 100
            for other in course.remaining()
            if other is not item
        )
        needed = (target - course.current_total() - others) / item.weight * 100
        forecasts.append(
            {
                "name": item.name,
                "low": prediction.low,
                "median": prediction.mean,
                "high": prediction.high,
                "sd": prediction.sd,
                "coursework": coursework,
                "coursework_known": bool(quiz_and_other),
                "midterm": midterm,
                "needed_for_a": needed,
            }
        )
    return forecasts


def sample_paths(
    course: Course,
    exam_models: model.Models,
    k: int = 25,
    pins: dict[str, float] | None = None,
    shift: float = 0.0,
    seed: int = 42,
) -> dict:
    """Running totals of ``k`` simulated semesters, item by item.

    Args:
        course: The course.
        exam_models: Trained final-exam models.
        k: How many semesters to sample.
        pins: Item name -> fixed score.
        shift: Points added to every random draw.
        seed: Random seed.

    Returns:
        Dict with ``names`` (all items in course order), ``fixed`` (True for
        taken or pinned items), ``scores`` and ``cumulative`` arrays
        (``cumulative`` has a leading 0 column), ``finals`` and ``grades``.
    """
    pins = pins or {}
    random_names, draws = simulate_items(course, exam_models, k, pins, shift, seed)
    # Build a k x m matrix of scores: fixed items repeat, random ones are drawn.
    scores = np.zeros((k, len(course.items)))
    fixed: list[bool] = []
    next_column = 0
    for position, item in enumerate(course.items):
        if item.name in pins:
            scores[:, position] = pins[item.name]
            fixed.append(True)
        elif item.score is not None:
            scores[:, position] = item.score
            fixed.append(True)
        else:
            # Random items appear in draws in the same (course) order.
            scores[:, position] = draws[:, next_column]
            next_column += 1
            fixed.append(False)
    weights = np.array([item.weight / 100 for item in course.items])
    # Points each item adds, then a running sum along the semester.
    contributions = scores * weights
    cumulative = np.hstack([np.zeros((k, 1)), np.cumsum(contributions, axis=1)])
    finals = cumulative[:, -1]
    return {
        "names": [item.name for item in course.items],
        "fixed": fixed,
        "scores": scores,
        "cumulative": cumulative,
        "finals": finals,
        "grades": letters_for(finals).tolist(),
    }


def convergence_snapshots(
    totals: np.ndarray, checkpoints: tuple[int, ...] = (10, 100, 1000, 10000)
) -> list[np.ndarray]:
    """The first N simulated totals for each checkpoint N.

    Args:
        totals: Simulated final totals, in the order they were drawn.
        checkpoints: Run counts to snapshot.

    Returns:
        One array per checkpoint (never longer than ``totals``).
    """
    return [totals[: min(count, len(totals))] for count in checkpoints]

