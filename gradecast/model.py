"""Gradient boosting model that predicts the final exam as a range.

Instead of one number, we train three "quantile" models (10th, 50th and 90th
percentile) so we get a likely range for the final exam. The training data is
the UCI Student Performance dataset, whose grades G1, G2, G3 (0-20 scale) are
mapped onto our course like this (an APPROXIMATION, not an exact match):

    coursework <- G1 * 5   (early-semester performance ~ quizzes/assignments)
    midterm    <- G2 * 5   (middle-of-semester grade ~ mid term exam)
    final      <- G3 * 5   (end-of-year grade ~ final exam; the target)

Two model sets exist so the app works before the midterm (coursework only)
and after it (coursework + midterm).
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# The three percentiles we predict: low end, middle, and high end.
QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)

# Which input columns each model set uses.
FEATURES: dict[str, list[str]] = {
    "with_midterm": ["coursework", "midterm"],
    "without_midterm": ["coursework"],
}

# The middle 80% of a normal distribution spans 2 * 1.2816 = 2.563 standard
# deviations, so (q90 - q10) / 2.563 converts our range into a standard deviation.
MIDDLE_80_WIDTH_IN_SD = 2.563

# Never claim to know the exam better than +/- 4 points; real exams are noisy.
MIN_EXAM_SD = 4.0

# Models are stored as {set name: {quantile: fitted model}}.
Models = dict[str, dict[float, HistGradientBoostingRegressor]]


@dataclass
class ExamPrediction:
    """Predicted distribution of a final exam score.

    Attributes:
        mean: The median (q50) prediction.
        sd: Spread, derived from the q10-q90 range (at least 4.0).
        low: The 10th percentile prediction.
        high: The 90th percentile prediction.
    """

    mean: float
    sd: float
    low: float
    high: float


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw UCI table into coursework / midterm / final columns.

    Args:
        raw: UCI table containing G1, G2 and G3 columns (0-20 scale).

    Returns:
        DataFrame with columns coursework, midterm, final on a 0-100 scale.
    """
    # Multiply by 5 to move from the 0-20 scale to our 0-100 scale.
    data = pd.DataFrame(
        {
            "coursework": raw["G1"] * 5.0,
            "midterm": raw["G2"] * 5.0,
            "final": raw["G3"] * 5.0,
        }
    )
    # A final of 0 after a non-zero midterm usually means the student dropped
    # out or never sat the exam. Those rows would teach the model that exams
    # randomly collapse to 0, so we drop them. Genuine 0/0 rows are kept.
    dropped_out = (data["final"] == 0) & (data["midterm"] > 0)
    return data[~dropped_out].reset_index(drop=True)


def train(df: pd.DataFrame) -> dict:
    """Train both model sets and hold out a test split.

    Args:
        df: Output of ``prepare_data``.

    Returns:
        Dict with ``models`` plus ``X_test`` / ``y_test`` / ``n_train`` for
        evaluation.
    """
    # 80/20 split with a fixed seed so results are reproducible.
    train_rows, test_rows = train_test_split(df, test_size=0.2, random_state=42)
    models: Models = {}
    for set_name, columns in FEATURES.items():
        models[set_name] = {}
        for quantile in QUANTILES:
            # loss="quantile" makes the model aim at the q-th percentile
            # instead of the average, so three models give a range.
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
            model.fit(train_rows[columns], train_rows["final"])
            models[set_name][quantile] = model
    return {
        "models": models,
        "X_test": test_rows[["coursework", "midterm"]],
        "y_test": test_rows["final"],
        "n_train": len(train_rows),
    }


def evaluate(models: Models, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Measure accuracy of each model set against a simple baseline.

    Args:
        models: Trained models.
        X_test: Test rows with coursework and midterm columns.
        y_test: True final scores for those rows.

    Returns:
        Per-set MAE, baseline MAE and q10-q90 coverage, plus test row count.
    """
    results: dict = {"n_test": int(len(y_test))}
    for set_name, columns in FEATURES.items():
        features = X_test[columns]
        low, mid, high = (models[set_name][q].predict(features) for q in QUANTILES)
        # Baseline: guess the final equals the average of what we know so far.
        baseline = features.mean(axis=1)
        results[set_name] = {
            "mae": float(np.mean(np.abs(mid - y_test))),
            "baseline_mae": float(np.mean(np.abs(baseline - y_test))),
            # Coverage: how often the true score falls inside q10..q90 (ideal 0.80).
            "coverage": float(np.mean((y_test >= low) & (y_test <= high))),
        }
    return results


def save(models: Models, path: str | Path) -> None:
    """Write trained models to disk.

    Args:
        models: Trained models.
        path: Destination file, e.g. models/gb_quantile.joblib.
    """
    joblib.dump(models, path)


def load(path: str | Path) -> Models:
    """Read trained models back from disk.

    Args:
        path: File written by ``save``.

    Returns:
        The same models dictionary.
    """
    return joblib.load(path)


def predict_exam(models: Models, coursework: float, midterm: float | None) -> ExamPrediction:
    """Predict the final exam distribution for one student.

    Args:
        models: Trained models.
        coursework: Average of the student's quiz/other scores (0-100).
        midterm: Mid term exam score, or None if not taken (uses the
            coursework-only models).

    Returns:
        An ExamPrediction, with scores kept within 0-100.

    Example:
        >>> predict_exam(models, 80, 75).mean  # doctest: +SKIP
        76.4
    """
    # Choose the model set that matches what the student has so far.
    if midterm is None:
        set_name, row = "without_midterm", {"coursework": [coursework]}
    else:
        set_name, row = "with_midterm", {"coursework": [coursework], "midterm": [midterm]}
    features = pd.DataFrame(row)
    # Sort the three outputs: separately trained quantile models can "cross"
    # (e.g. q10 above q50), which would be nonsense for a range.
    low, mid, high = sorted(
        float(models[set_name][q].predict(features)[0]) for q in QUANTILES
    )
    sd = max((high - low) / MIDDLE_80_WIDTH_IN_SD, MIN_EXAM_SD)
    return ExamPrediction(
        mean=float(np.clip(mid, 0, 100)),
        sd=sd,
        low=float(np.clip(low, 0, 100)),
        high=float(np.clip(high, 0, 100)),
    )
