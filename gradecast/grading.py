"""Universitas Indonesia undergraduate grading table.

This module turns a numeric course score (0-100) into a letter grade and an
"indeks" (grade points), using the official table from Peraturan Rektor UI
No. 1/2024. The lower bound of each band is inclusive, so 85.00 is an A but
84.99 is an A-. It has no Streamlit code so it can be reused anywhere.
"""

import numpy as np

# Each row is (minimum score, letter, indeks), ordered from the highest band
# to the lowest. Ordering matters: we look for the first band the score fits.
GRADE_TABLE: list[tuple[float, str, float]] = [
    (85.0, "A", 4.00),
    (80.0, "A-", 3.70),
    (75.0, "B+", 3.30),
    (70.0, "B", 3.00),
    (65.0, "B-", 2.70),
    (60.0, "C+", 2.30),
    (55.0, "C", 2.00),
    (40.0, "D", 1.00),
    (0.0, "E", 0.00),
]

# Just the letters, high -> low. Used for chart axes and result ordering.
LETTERS: list[str] = [letter for _, letter, _ in GRADE_TABLE]

# Parallel arrays used by the vectorised lookup further below.
_MIN_SCORES = np.array([row[0] for row in GRADE_TABLE])
_INDEKS_VALUES = np.array([row[2] for row in GRADE_TABLE])


def _band_position(score: float) -> int:
    """Return the row number in GRADE_TABLE that a score falls into.

    Args:
        score: A course score, normally between 0 and 100.

    Returns:
        Index into GRADE_TABLE (0 = A, last = E).

    Example:
        >>> _band_position(90)
        0
    """
    # Walk down the table and stop at the first band whose minimum we reach.
    for position, (minimum, _, _) in enumerate(GRADE_TABLE):
        if score >= minimum:
            return position
    # Anything below 0 still counts as the lowest band (E).
    return len(GRADE_TABLE) - 1


def to_letter(score: float) -> str:
    """Convert a numeric score to its letter grade.

    Args:
        score: Course score from 0 to 100.

    Returns:
        The letter, e.g. "A-".

    Example:
        >>> to_letter(84.99)
        'A-'
    """
    return GRADE_TABLE[_band_position(score)][1]


def to_indeks(score: float) -> float:
    """Convert a numeric score to its indeks (grade points on a 4.00 scale).

    Args:
        score: Course score from 0 to 100.

    Returns:
        The indeks, e.g. 3.70 for an A-.

    Example:
        >>> to_indeks(85)
        4.0
    """
    return GRADE_TABLE[_band_position(score)][2]


def letters_for(scores: np.ndarray) -> np.ndarray:
    """Convert many scores to letters at once (fast, no Python loop).

    Args:
        scores: Array of course scores, e.g. 10,000 simulated totals.

    Returns:
        Array of letter strings with the same length as ``scores``.

    Example:
        >>> letters_for(np.array([90.0, 50.0])).tolist()
        ['A', 'D']
    """
    # np.select picks, for every score, the letter of the first condition that
    # is true. Conditions are listed high -> low, matching GRADE_TABLE order.
    conditions = [scores >= minimum for minimum in _MIN_SCORES[:-1]]
    # Anything that matches no condition (below 40) falls through to "E".
    return np.select(conditions, LETTERS[:-1], default=LETTERS[-1])
