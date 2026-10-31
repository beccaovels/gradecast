"""Bayesian prediction for quizzes, participation and other non-exam items.

Bayesian updating in plain words: start from what a typical student scores
(the prior), then let each of your own scores pull that belief toward you (the
evidence). With few scores the prior matters a lot; with many, your own data
dominates. The result is a "belief" about one future score: a mean and a
spread, which the simulator then draws random scores from.
"""

import math
from dataclasses import dataclass


@dataclass
class Belief:
    """What we expect for ONE future score.

    Attributes:
        mean: Expected score.
        sd: Standard deviation of the predictive distribution, i.e. how far a
            single future score is likely to land from the mean.
    """

    mean: float
    sd: float


def _noise_estimate(scores: list[float]) -> float:
    """Guess how much a student's individual scores bounce around.

    Args:
        scores: The student's scores so far.

    Returns:
        A noise standard deviation to use in the update.
    """
    # With 0 or 1 score we cannot measure the spread, so assume a typical 10.
    if len(scores) < 2:
        return 10.0
    mean = sum(scores) / len(scores)
    # Sample standard deviation (divide by n - 1 because the mean is estimated).
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    # Floor at 6.0: a few similar scores in a row (one lucky streak) must not
    # fool us into thinking the student is perfectly predictable.
    return max(math.sqrt(variance), 6.0)


def update(
    scores: list[float],
    prior_mean: float = 75.0,
    prior_sd: float = 12.0,
    noise_sd: float | None = None,
) -> Belief:
    """Update the prior belief with the student's own scores.

    This is the normal-normal conjugate update with known observation noise:

        precision_prior     = 1 / prior_sd^2
        precision_data      = n / noise_sd^2
        precision_posterior = precision_prior + precision_data
        mean_posterior      = (prior_mean / prior_sd^2 + sum(scores) / noise_sd^2)
                              / precision_posterior
        predictive_sd       = sqrt(noise_sd^2 + 1 / precision_posterior)

    The predictive sd adds the noise of one new score to our remaining
    uncertainty about the student's true level.

    Args:
        scores: The student's scores so far (may be empty).
        prior_mean: Typical score before seeing any data.
        prior_sd: How much students' true levels vary around that typical score.
        noise_sd: Score-to-score noise; None = estimate it from ``scores``.

    Returns:
        A Belief about the next score.

    Example:
        >>> update([90, 92, 88]).mean > 75
        True
    """
    # Decide the noise level, either from the caller or from the data.
    noise = noise_sd if noise_sd is not None else _noise_estimate(scores)
    # Precision = 1 / variance; precisions simply add up in a normal update.
    prior_precision = 1 / prior_sd**2
    data_precision = len(scores) / noise**2
    posterior_precision = prior_precision + data_precision
    # Posterior mean is a precision-weighted average of prior and data.
    posterior_mean = (prior_mean * prior_precision + sum(scores) / noise**2) / posterior_precision
    # Predictive sd: noise of one new score plus our uncertainty about the mean.
    predictive_sd = math.sqrt(noise**2 + 1 / posterior_precision)
    return Belief(mean=posterior_mean, sd=predictive_sd)
