"""Probability and treatment-effect metrics, written out so each one can be checked by hand."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Z95 = 1.959963984540054
EPS = 1e-15


def _check_probs(y: npt.ArrayLike, p: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate 0/1 labels and probabilities of equal, non-zero length."""
    ya = np.asarray(y, dtype=np.float64)
    pa = np.asarray(p, dtype=np.float64)
    if ya.ndim != 1 or ya.shape != pa.shape or ya.size == 0:
        raise ValueError("labels and probabilities must be 1-D, non-empty and the same length")
    if not np.isin(ya, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(pa).all() or (pa < 0).any() or (pa > 1).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    return ya, pa


def brier(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Mean squared error between probabilities and 0/1 outcomes."""
    ya, pa = _check_probs(y, p)
    return float(np.mean((pa - ya) ** 2))


def log_loss(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Mean negative log-likelihood, with probabilities clipped away from 0 and 1."""
    ya, pa = _check_probs(y, p)
    pc = np.clip(pa, EPS, 1 - EPS)
    return float(-np.mean(ya * np.log(pc) + (1 - ya) * np.log(1 - pc)))


@dataclass(frozen=True)
class Bin:
    """One reliability bin: probability range, size, mean prediction and observed rate."""

    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


def reliability(y: npt.ArrayLike, p: npt.ArrayLike, bins: int = 10) -> list[Bin]:
    """Equal-width reliability bins over [0, 1]; empty bins are left out."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    ya, pa = _check_probs(y, p)
    # Bin i holds [i/bins, (i+1)/bins); probability 1.0 goes into the last bin.
    idx = np.minimum((pa * bins).astype(np.int64), bins - 1)
    out = []
    for i in range(bins):
        mask = idx == i
        n = int(mask.sum())
        if n:
            out.append(
                Bin(i / bins, (i + 1) / bins, n, float(pa[mask].mean()), float(ya[mask].mean()))
            )
    return out


def ece(y: npt.ArrayLike, p: npt.ArrayLike, bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted gap between prediction and outcome rate."""
    table = reliability(y, p, bins)
    n = sum(b.count for b in table)
    return float(sum(b.count / n * abs(b.mean_predicted - b.observed_rate) for b in table))


def auc(y: npt.ArrayLike, score: npt.ArrayLike) -> float:
    """Area under the ROC curve from average ranks (ties count one half)."""
    ya = np.asarray(y, dtype=np.float64)
    sa = np.asarray(score, dtype=np.float64)
    if ya.ndim != 1 or ya.shape != sa.shape or not np.isfinite(sa).all():
        raise ValueError("labels and scores must be 1-D, finite and the same length")
    if not np.isin(ya, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    n_pos = int(ya.sum())
    n_neg = ya.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(sa, kind="mergesort")
    ranks = np.empty(sa.size, dtype=np.float64)
    sorted_s = sa[order]
    i = 0
    while i < sa.size:
        j = i
        while j + 1 < sa.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = float(ranks[ya == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass(frozen=True)
class Effect:
    """Difference in outcome rates between treated and control, with a 95% normal interval."""

    estimate: float
    low: float
    high: float
    n_treated: int
    n_control: int


def diff_in_means(t: npt.ArrayLike, y: npt.ArrayLike) -> Effect:
    """Average treatment effect estimate from randomized data (treated minus control mean)."""
    ta = np.asarray(t)
    ya = np.asarray(y, dtype=np.float64)
    if ta.ndim != 1 or ta.shape != ya.shape:
        raise ValueError("treatment and outcome must be 1-D and the same length")
    if not np.isin(ta, (0, 1)).all():
        raise ValueError("treatment must be 0 or 1")
    y1, y0 = ya[ta == 1], ya[ta == 0]
    if y1.size < 2 or y0.size < 2:
        raise ValueError("each group needs at least two customers")
    est = float(y1.mean() - y0.mean())
    se = math.sqrt(float(y1.var(ddof=1)) / y1.size + float(y0.var(ddof=1)) / y0.size)
    return Effect(est, est - Z95 * se, est + Z95 * se, int(y1.size), int(y0.size))


def smd(x: npt.ArrayLike, t: npt.ArrayLike) -> FloatArray:
    """Standardized mean difference of each feature between treated and control."""
    xa = np.asarray(x, dtype=np.float64)
    ta = np.asarray(t)
    if xa.ndim != 2 or ta.ndim != 1 or xa.shape[0] != ta.size:
        raise ValueError("x must be 2-D with one row per treatment flag")
    if not np.isin(ta, (0, 1)).all():
        raise ValueError("treatment must be 0 or 1")
    x1, x0 = xa[ta == 1], xa[ta == 0]
    if x1.shape[0] < 2 or x0.shape[0] < 2:
        raise ValueError("each group needs at least two customers")
    pooled = np.sqrt((x1.var(axis=0, ddof=1) + x0.var(axis=0, ddof=1)) / 2)
    diff = x1.mean(axis=0) - x0.mean(axis=0)
    # Spreads and gaps below rounding noise of the feature's scale count as exactly zero.
    tol = 1e-9 * np.maximum(1.0, np.abs(xa).max(axis=0))
    diff = np.where(np.abs(diff) <= tol, 0.0, diff)
    spread = pooled > tol
    # Constant within both groups: no imbalance if the constants agree, total imbalance if not.
    safe = np.where(spread, pooled, 1.0)
    degenerate = np.where(diff == 0, 0.0, np.copysign(np.inf, diff))
    return np.where(spread, diff / safe, degenerate).astype(np.float64)
