"""Distribution distances used by the Phase 3 gap profile.

Every distance is reported on its own scale. Lower means the two samples are
closer. One-dimensional Wasserstein-1 is the L1 distance between quantile
functions. Class frequencies use base-2 Jensen–Shannon divergence, which lies
in ``[0, 1]``. Flow direction uses the circular Wasserstein-1 distance in
radians, so ``-pi`` and ``pi`` are the same direction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(wasserstein_distance(left, right))


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Base-2 Jensen–Shannon divergence. Inputs are non-negative counts or probabilities."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"JS support mismatch: {p.shape} vs {q.shape}")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def circular_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    """W1 on a circle, in radians. ``left`` and ``right`` are histogram counts on equal bins."""
    p = np.asarray(left, dtype=np.float64).reshape(-1)
    q = np.asarray(right, dtype=np.float64).reshape(-1)
    if p.shape != q.shape or p.size == 0:
        raise ValueError("Circular histograms must share one non-empty set of bins")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    cdf = np.cumsum(p - q)
    alpha = float(np.median(cdf))
    bin_width = 2.0 * np.pi / p.size
    return float(np.sum(np.abs(cdf - alpha)) * bin_width)


def class_counts(labels: np.ndarray, support: np.ndarray) -> np.ndarray:
    counts = np.zeros(support.size, dtype=np.float64)
    if labels.size == 0:
        return counts
    index = {int(label): i for i, label in enumerate(support.tolist())}
    for label in labels.tolist():
        slot = index.get(int(label))
        if slot is not None:
            counts[slot] += 1
    return counts
