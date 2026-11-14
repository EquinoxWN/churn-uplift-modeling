"""Synthetic customers with known churn probabilities and known offer effects, for tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .data import CampaignData, ChurnData

FloatArray = npt.NDArray[np.float64]
SEGMENTS = ("persuadable", "sure_thing", "lost_cause", "do_not_disturb")


def _sigmoid(z: FloatArray) -> FloatArray:
    """Logistic function."""
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


@dataclass(frozen=True)
class SyntheticChurn:
    """Churn data plus the true churn probability of every customer."""

    data: ChurnData
    true_p: FloatArray


@dataclass(frozen=True)
class SyntheticCampaign:
    """Randomized campaign data plus each customer's true effect and planted segment."""

    data: CampaignData
    true_uplift: FloatArray
    segment: npt.NDArray[np.str_]


def make_churn(n: int = 5000, seed: int = 0) -> SyntheticChurn:
    """Customers whose churn probability is a known non-linear function of six features."""
    if n < 10:
        raise ValueError("n must be at least 10")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    logit = -1.2 + 1.0 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3] + 0.6 * (x[:, 4] > 0.5)
    p = _sigmoid(logit)
    y = (rng.random(n) < p).astype(np.int64)
    names = tuple(f"x{i}" for i in range(6))
    return SyntheticChurn(ChurnData(x=x, y=y, names=names), p)


def make_campaign(n: int = 20000, seed: int = 0, treated_share: float = 0.5) -> SyntheticCampaign:
    """Randomized offer data with four planted segments and a known effect per customer."""
    if n < 10:
        raise ValueError("n must be at least 10")
    if not 0.0 < treated_share < 1.0:
        raise ValueError("treated_share must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    # Baseline response between 0.05 and 0.75, so every planted effect stays a valid probability.
    p0 = 0.05 + 0.70 * _sigmoid(-0.3 + 1.5 * x[:, 0])
    # Persuadables (x2 > 0.5) gain +0.10 to +0.25; 'do not disturb' (x3 > 1) lose half their response.
    persuadable = x[:, 2] > 0.5
    do_not_disturb = (x[:, 3] > 1.0) & ~persuadable
    effect = np.where(persuadable, 0.10 + 0.15 * _sigmoid(3.0 * (x[:, 2] - 0.5)), 0.0)
    effect = np.where(do_not_disturb, -0.5 * p0, effect)
    p1 = p0 + effect
    true_uplift = effect
    t = (rng.random(n) < treated_share).astype(np.int64)
    u = rng.random(n)
    y = np.where(t == 1, u < p1, u < p0).astype(np.int64)
    segment = np.where(
        persuadable,
        "persuadable",
        np.where(do_not_disturb, "do_not_disturb", np.where(p0 >= 0.4, "sure_thing", "lost_cause")),
    )
    names = tuple(f"x{i}" for i in range(6))
    return SyntheticCampaign(CampaignData(x=x, t=t, y=y, names=names), true_uplift, segment)
