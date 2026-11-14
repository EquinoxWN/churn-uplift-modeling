"""Metrics on worked examples, their limits, and refused inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from churn_uplift_modeling.metrics import (
    auc,
    brier,
    diff_in_means,
    ece,
    log_loss,
    reliability,
    smd,
)


def test_brier_and_log_loss_on_a_worked_example() -> None:
    y = [1, 0, 1, 0]
    p = [0.9, 0.2, 0.6, 0.5]
    assert brier(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.25) / 4)
    expected = -(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.5)) / 4
    assert log_loss(y, p) == pytest.approx(expected)
    # A confident wrong answer is clipped, not infinite.
    assert math.isfinite(log_loss([1], [0.0]))


def test_reliability_bins_and_ece_on_a_worked_example() -> None:
    y = [0, 0, 1, 1, 1, 0]
    p = [0.05, 0.15, 0.15, 0.85, 0.95, 1.0]
    bins = reliability(y, p, bins=2)
    assert [(b.low, b.high, b.count) for b in bins] == [(0.0, 0.5, 3), (0.5, 1.0, 3)]
    assert bins[0].mean_predicted == pytest.approx(0.35 / 3)
    assert bins[0].observed_rate == pytest.approx(1 / 3)
    expected = 0.5 * abs(0.35 / 3 - 1 / 3) + 0.5 * abs(2.8 / 3 - 2 / 3)
    assert ece(y, p, bins=2) == pytest.approx(expected)


def test_ece_is_zero_when_rates_match_and_large_when_overconfident() -> None:
    y = [1, 0, 1, 0, 1, 1, 1, 0]
    assert ece(y, [0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75]) == pytest.approx(0.0)
    assert ece([1, 0, 1, 0], [0.99, 0.99, 0.99, 0.99]) == pytest.approx(0.49)


def test_probability_one_falls_into_the_last_bin() -> None:
    bins = reliability([1, 1], [1.0, 0.95], bins=10)
    assert len(bins) == 1
    assert bins[0].count == 2
    assert bins[0].low == pytest.approx(0.9)


def test_auc_on_a_worked_example_with_ties() -> None:
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert auc([0, 0, 1], [0.1, 0.2, 0.9]) == pytest.approx(1.0)
    assert auc([1, 1, 0], [0.1, 0.2, 0.9]) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 1)  # many ties
    pairs = [(a, b) for a in s[y == 1] for b in s[y == 0]]
    brute = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
    assert auc(y, s) == pytest.approx(brute)


@pytest.mark.parametrize(
    ("y", "p"),
    [
        ([], []),
        ([1, 0], [0.5]),
        ([2, 0], [0.5, 0.5]),
        ([1, 0], [1.5, 0.5]),
        ([1, 0], [math.nan, 0.5]),
    ],
)
def test_probability_metrics_refuse_bad_input(y: list[float], p: list[float]) -> None:
    with pytest.raises(ValueError, match=r"labels|probabilities"):
        brier(y, p)
    with pytest.raises(ValueError, match=r"labels|probabilities"):
        ece(y, p)


def test_auc_and_bins_refuse_bad_input() -> None:
    with pytest.raises(ValueError, match="both classes"):
        auc([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError, match="finite"):
        auc([1, 0], [math.inf, 0.3])
    with pytest.raises(ValueError, match="0 or 1"):
        auc([3, 0], [0.1, 0.3])
    with pytest.raises(ValueError, match="bins"):
        reliability([1], [0.5], bins=0)


def test_diff_in_means_and_its_interval() -> None:
    t = [1, 1, 1, 1, 0, 0, 0, 0]
    y = [1, 1, 1, 0, 1, 0, 0, 0]
    e = diff_in_means(t, y)
    assert e.estimate == pytest.approx(0.5)
    se = math.sqrt(0.25 / 4 + 0.25 / 4)  # sample variance 1/4 in each group
    assert e.low == pytest.approx(0.5 - 1.959963984540054 * se)
    assert e.high == pytest.approx(0.5 + 1.959963984540054 * se)
    assert (e.n_treated, e.n_control) == (4, 4)
    with pytest.raises(ValueError, match="at least two"):
        diff_in_means([1, 0, 0], [1, 0, 1])
    with pytest.raises(ValueError, match="0 or 1"):
        diff_in_means([2, 0], [1, 0])
    with pytest.raises(ValueError, match="same length"):
        diff_in_means([1, 0], [1])


def test_smd_finds_imbalance_and_ignores_constant_features() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0], [0.0, 5.0], [2.0, 5.0]])
    t = np.array([1, 1, 0, 0])
    d = smd(x, t)
    # Means 2 and 1, sample variances 2 and 2: SMD = 1 / sqrt(2).
    assert d[0] == pytest.approx(1 / math.sqrt(2))
    assert d[1] == 0.0
    split = smd(np.array([[1.0], [1.0], [0.0], [0.0]]), t)
    assert split[0] == math.inf
    with pytest.raises(ValueError, match="2-D"):
        smd(np.array([1.0, 2.0]), t[:2])
    with pytest.raises(ValueError, match="at least two"):
        smd(x[:3], np.array([1, 0, 0]))
