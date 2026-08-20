"""Metric calculation for Zoom-IQA score regression."""

from __future__ import annotations

from array import array
import math
from typing import Iterable


def _float32(values: Iterable[float]) -> list[float]:
    return list(array("f", (float(value) for value in values)))


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = math.fsum(a * b for a, b in zip(left_centered, right_centered))
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    return numerator / denominator if denominator else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def fit_isotonic(predictions: list[float], targets: list[float]) -> list[float]:
    """Map predictions onto the target scale with monotonic isotonic regression.

    Predictions carry only two decimals, so the same value repeats many times in
    a benchmark run. ``IsotonicRegression`` pools tied inputs by weight before
    fitting, which a naive pool-adjacent-violators pass does not do, so the
    reference implementation is used rather than a local reimplementation.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as error:  # No unfitted fallback exists, so fail loudly.
        raise RuntimeError(
            "scikit-learn is required to compute fit_plcc/fit_srcc; "
            "install it with `python -m pip install scikit-learn`"
        ) from error
    regressor = IsotonicRegression(increasing=True, out_of_bounds="clip")
    regressor.fit(predictions, targets)
    return _float32(regressor.predict(predictions))


def correlations(
    predictions: Iterable[float], targets: Iterable[float]
) -> dict[str, float | None]:
    """PLCC/SRCC after isotonic rescaling onto the ground-truth scale."""
    prediction_array = _float32(predictions)
    target_array = _float32(targets)

    if len(prediction_array) != len(target_array):
        raise ValueError("prediction and target counts differ")
    if len(prediction_array) < 2:
        raise ValueError("at least two valid predictions are required")
    fitted = fit_isotonic(prediction_array, target_array)
    return {
        "fit_plcc": _pearson(fitted, target_array),
        "fit_srcc": _pearson(_rank(fitted), _rank(target_array)),
    }
