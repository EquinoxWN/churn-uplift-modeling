"""Uplift meta-learners (S, T, X) on randomized campaign data, with a randomization check."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CampaignData
from .metrics import Effect, diff_in_means, smd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
LEARNERS = ("s", "t", "x")
BASES = ("logistic", "gbm")
MAX_SMD = 0.1
MIN_PER_CELL = 5


def make_models(base: str, seed: int = 0) -> tuple[Any, Any]:
    """An unfitted (classifier, regressor) pair for one base family."""
    if base == "logistic":
        return (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        )
    if base == "gbm":
        params: dict[str, Any] = {
            "max_iter": 150,
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "random_state": seed,
        }
        return HistGradientBoostingClassifier(**params), HistGradientBoostingRegressor(**params)
    raise ValueError(f"base must be one of {BASES}, got {base!r}")


def check_campaign(x: FloatArray, t: IntArray, y: IntArray) -> None:
    """Validate features, 0/1 treatment and 0/1 outcomes, with every arm-outcome cell present."""
    if x.ndim != 2 or t.ndim != 1 or y.ndim != 1 or not x.shape[0] == t.size == y.size:
        raise ValueError("x must be 2-D with one treatment flag and one outcome per row")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if not np.isin(t, (0, 1)).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("treatment and outcome must be 0 or 1")
    for arm in (0, 1):
        for out in (0, 1):
            if int(((t == arm) & (y == out)).sum()) < MIN_PER_CELL:
                raise ValueError(
                    f"need at least {MIN_PER_CELL} customers with treatment={arm}, outcome={out}"
                )


@dataclass(frozen=True)
class Balance:
    """Randomization check: treated share and the largest standardized mean difference."""

    treated_share: float
    max_abs_smd: float
    worst_feature: str
    balanced: bool


def check_randomized(data: CampaignData, max_smd: float = MAX_SMD) -> Balance:
    """Compare feature means between arms; large gaps mean the offer was not randomized."""
    d = smd(data.x, data.t)
    worst = int(np.argmax(np.abs(d)))
    value = float(abs(d[worst]))
    return Balance(float(data.t.mean()), value, data.names[worst], value <= max_smd)


class SLearner:
    """One classifier on features plus the treatment flag; uplift = p(x, 1) - p(x, 0)."""

    name = "s"

    def __init__(self, base: str = "logistic", seed: int = 0) -> None:
        """Choose the base model family."""
        self.model: Any = make_models(base, seed)[0]
        self._fitted = False

    def fit(self, x: FloatArray, t: IntArray, y: IntArray) -> SLearner:
        """Fit on all customers with the treatment flag as an extra column."""
        check_campaign(x, t, y)
        self.model.fit(np.column_stack([x, t]), y)
        self._fitted = True
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Estimated change in outcome probability if the customer gets the offer."""
        _check_fitted(self._fitted, x)
        ones, zeros = np.ones(x.shape[0]), np.zeros(x.shape[0])
        p1 = self.model.predict_proba(np.column_stack([x, ones]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([x, zeros]))[:, 1]
        return np.asarray(p1 - p0, dtype=np.float64)


class TLearner:
    """Separate classifiers for treated and control; uplift = p1(x) - p0(x)."""

    name = "t"

    def __init__(self, base: str = "logistic", seed: int = 0) -> None:
        """Choose the base model family."""
        clf, _ = make_models(base, seed)
        self.m1: Any = clone(clf)
        self.m0: Any = clone(clf)
        self._fitted = False

    def fit(self, x: FloatArray, t: IntArray, y: IntArray) -> TLearner:
        """Fit one model per arm."""
        check_campaign(x, t, y)
        self.m1.fit(x[t == 1], y[t == 1])
        self.m0.fit(x[t == 0], y[t == 0])
        self._fitted = True
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Estimated change in outcome probability if the customer gets the offer."""
        _check_fitted(self._fitted, x)
        p1 = self.m1.predict_proba(x)[:, 1]
        p0 = self.m0.predict_proba(x)[:, 1]
        return np.asarray(p1 - p0, dtype=np.float64)


class XLearner:
    """Kuenzel et al. (2019): impute each arm's effects with the other arm's model, then blend."""

    name = "x"

    def __init__(self, base: str = "logistic", seed: int = 0) -> None:
        """Choose the base model family."""
        clf, reg = make_models(base, seed)
        self.m1: Any = clone(clf)
        self.m0: Any = clone(clf)
        self.tau1: Any = clone(reg)
        self.tau0: Any = clone(reg)
        self.propensity = 0.5
        self._fitted = False

    def fit(self, x: FloatArray, t: IntArray, y: IntArray) -> XLearner:
        """Stage 1 outcome models per arm, stage 2 effect models on imputed effects."""
        check_campaign(x, t, y)
        x1, y1 = x[t == 1], y[t == 1].astype(np.float64)
        x0, y0 = x[t == 0], y[t == 0].astype(np.float64)
        self.m1.fit(x1, y[t == 1])
        self.m0.fit(x0, y[t == 0])
        # Treated: observed minus predicted control outcome; control: predicted treated minus observed.
        self.tau1.fit(x1, y1 - self.m0.predict_proba(x1)[:, 1])
        self.tau0.fit(x0, self.m1.predict_proba(x0)[:, 1] - y0)
        # Randomized data: the propensity is the constant treated share.
        self.propensity = float(t.mean())
        self._fitted = True
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Propensity-weighted blend of the two effect models."""
        _check_fitted(self._fitted, x)
        g = self.propensity
        tau = g * self.tau0.predict(x) + (1 - g) * self.tau1.predict(x)
        return np.asarray(np.clip(tau, -1.0, 1.0), dtype=np.float64)


Learner = SLearner | TLearner | XLearner


def _check_fitted(fitted: bool, x: FloatArray) -> None:
    """Refuse predictions before fit or on malformed input."""
    if not fitted:
        raise RuntimeError("fit the learner before predicting")
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite 2-D array")


def make_learner(name: str, base: str = "logistic", seed: int = 0) -> Learner:
    """Build an unfitted learner by short name (s, t or x)."""
    if name == "s":
        return SLearner(base, seed)
    if name == "t":
        return TLearner(base, seed)
    if name == "x":
        return XLearner(base, seed)
    raise ValueError(f"learner must be one of {LEARNERS}, got {name!r}")


@dataclass(frozen=True)
class UpliftScore:
    """Holdout results of one learner."""

    learner: str
    base: str
    mean_predicted: float
    observed: Effect
    top_share: float
    top_observed: Effect
    rest_observed: Effect
    corr_true: float | None
    rmse_true: float | None
    fit_seconds: float


def evaluate_uplift(
    data: CampaignData,
    learners: tuple[str, ...] = LEARNERS,
    base: str = "logistic",
    *,
    true_uplift: FloatArray | None = None,
    test_size: float = 0.3,
    top_share: float = 0.3,
    seed: int = 0,
) -> list[UpliftScore]:
    """Fit each learner on one split; compare its predictions with observed holdout effects."""
    unknown = [n for n in learners if n not in LEARNERS]
    if unknown or not learners:
        raise ValueError(f"unknown or missing learners: {unknown}")
    if base not in BASES:
        raise ValueError(f"base must be one of {BASES}, got {base!r}")
    if not 0.0 < test_size < 1.0 or not 0.0 < top_share < 1.0:
        raise ValueError("test_size and top_share must be strictly between 0 and 1")
    check_campaign(data.x, data.t, data.y)
    if true_uplift is not None and true_uplift.shape != data.t.shape:
        raise ValueError("true_uplift needs one value per customer")
    idx = np.arange(data.t.size)
    # Stratify on arm and outcome so both splits keep every cell.
    tr, te = train_test_split(
        idx, test_size=test_size, stratify=data.t * 2 + data.y, random_state=seed
    )
    x_te, t_te, y_te = data.x[te], data.t[te], data.y[te]
    out = []
    for name in learners:
        start = time.perf_counter()
        learner = make_learner(name, base, seed).fit(data.x[tr], data.t[tr], data.y[tr])
        elapsed = time.perf_counter() - start
        pred = learner.predict(x_te)
        # Highest predicted uplift first; stable sort keeps ties in holdout order.
        order = np.argsort(-pred, kind="stable")
        k = max(round(top_share * pred.size), 1)
        top, rest = order[:k], order[k:]
        corr = rmse = None
        if true_uplift is not None:
            truth = true_uplift[te]
            corr = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 0 else 0.0
            rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        out.append(
            UpliftScore(
                learner=name,
                base=base,
                mean_predicted=float(pred.mean()),
                observed=diff_in_means(t_te, y_te),
                top_share=top_share,
                top_observed=diff_in_means(t_te[top], y_te[top]),
                rest_observed=diff_in_means(t_te[rest], y_te[rest]),
                corr_true=corr,
                rmse_true=rmse,
                fit_seconds=elapsed,
            )
        )
    return out
