"""Sensor calibration against ground truth (the first "stop and tune" step).

Robust statistics everywhere (median, MAD): a few outliers (a missed association, a
ghost) must not move the calibration. MAD * 1.4826 estimates the standard deviation of
the well-behaved part.

Every function takes plain arrays, so it is unit-tested on synthetic numbers.
"""

import math

import numpy as np

MAD_TO_SIGMA = 1.4826


def robust_stats(x):
    """(median, robust sigma, n) of the finite values of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan, 0
    med = float(np.median(x))
    return med, float(MAD_TO_SIGMA * np.median(np.abs(x - med))), len(x)


def binned(distance, values, edges):
    """Per distance bin: (center, median, robust sigma, n) for bins with >= 10 samples."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (distance >= lo) & (distance < hi) & np.isfinite(values)
        if sel.sum() >= 10:
            med, sig, n = robust_stats(values[sel])
            rows.append(((lo + hi) / 2.0, med, sig, n))
    return rows


def fit_linear_sigma(rows, floor=0.02):
    """Fit sigma = a + b * distance through binned robust sigmas (least squares, a, b >= 0)."""
    if len(rows) < 2:
        return (rows[0][2], 0.0) if rows else (math.nan, math.nan)
    d = np.array([r[0] for r in rows])
    s = np.array([r[2] for r in rows])
    b, a = np.polyfit(d, s, 1)
    if b < 0:
        a, b = float(np.median(s)), 0.0
    return max(float(a), floor), max(float(b), 0.0)


def sign_agreement(measured, truth, min_abs=0.5):
    """Fraction of frames where measured and true range-rate have the same sign (|truth| > min_abs)."""
    ok = np.isfinite(measured) & np.isfinite(truth) & (np.abs(truth) > min_abs)
    if not ok.any():
        return math.nan
    return float(np.mean(np.sign(measured[ok]) == np.sign(truth[ok])))


def sigma_scale(distance, error, sigma, edges, min_distance=8.0, limits=(0.3, 3.0)):
    """How much the reported sigma must be scaled to match the actual spread of the error.

    Uses normalized errors z = (error - bias of its distance bin) / reported sigma, beyond
    min_distance (close to the lead the error is dominated by the constant car-width
    uncertainty, which is a bias, not noise). Returns (scale, n) with scale clamped to limits.
    """
    ok = np.isfinite(distance) & np.isfinite(error) & np.isfinite(sigma) & (sigma > 0) & (distance >= min_distance)
    z = np.full(len(error), np.nan)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (distance >= lo) & (distance < hi)
        if sel.sum() >= 5:
            z[sel] = (error[sel] - np.median(error[sel])) / sigma[sel]
    _, spread, n = robust_stats(z)
    if n < 20 or not math.isfinite(spread):
        return math.nan, n
    return float(np.clip(spread, *limits)), n
