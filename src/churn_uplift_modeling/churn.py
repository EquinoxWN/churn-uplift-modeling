"""Churn risk models with cross-validated probability calibration, scored on a holdout."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ChurnData
from .metrics import Bin, auc, brier, ece, log_loss, reliability

FloatArray = npt.NDArray[np.float64]
BASES = ("logistic", "gbm", "naive_bayes")
CALIBRATIONS = ("none", "isotonic", "sigmoid")


def make_classifier(base: str, seed: int = 0) -> Any:
    """A fresh scikit-learn classifier for one of the supported base models."""
    if base == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if base == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=seed,
        )
    if base == "naive_bayes":
        return GaussianNB()
    raise ValueError(f"base must be one of {BASES}, got {base!r}")


def _check_xy(x: FloatArray, y: npt.NDArray[np.int64]) -> None:
    """Validate a feature matrix and 0/1 labels."""
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("x must be 2-D with one row per label")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")


@dataclass
class ChurnModel:
    """A fitted churn classifier whose predict_proba returns P(churn) per customer."""

    base: str
    calibration: str
    estimator: Any

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """Probability that each customer churns."""
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("features must be a finite 2-D array")
        return np.asarray(self.estimator.predict_proba(x)[:, 1], dtype=np.float64)


def fit_churn(
    x: FloatArray,
    y: npt.NDArray[np.int64],
    base: str = "logistic",
    calibration: str = "isotonic",
    *,
    folds: int = 5,
    seed: int = 0,
) -> ChurnModel:
    """Fit a base classifier, optionally calibrated by cross-fitting on held-out folds."""
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CALIBRATIONS}, got {calibration!r}")
    _check_xy(x, y)
    minority = int(min(y.sum(), y.size - y.sum()))
    if minority < max(folds, 2):
        raise ValueError(f"need at least {max(folds, 2)} customers in each class")
    clf = make_classifier(base, seed)
    if calibration == "none":
        clf.fit(x, y)
        return ChurnModel(base, calibration, clf)
    # Each fold's calibrator is fitted on data its classifier did not see.
    cal = CalibratedClassifierCV(clf, method=calibration, cv=folds, ensemble=True)
    cal.fit(x, y)
    return ChurnModel(base, calibration, cal)


@dataclass(frozen=True)
class ChurnScore:
    """Holdout scores of one fitted churn model."""

    base: str
    calibration: str
    auc: float
    brier: float
    log_loss: float
    ece: float
    mean_predicted: float
    observed_rate: float
    fit_seconds: float
    bins: tuple[Bin, ...]


def score(
    model: ChurnModel, x: FloatArray, y: npt.NDArray[np.int64], fit_seconds: float
) -> ChurnScore:
    """Score a fitted model on holdout data."""
    p = model.predict_proba(x)
    return ChurnScore(
        base=model.base,
        calibration=model.calibration,
        auc=auc(y, p),
        brier=brier(y, p),
        log_loss=log_loss(y, p),
        ece=ece(y, p),
        mean_predicted=float(p.mean()),
        observed_rate=float(y.mean()),
        fit_seconds=fit_seconds,
        bins=tuple(reliability(y, p)),
    )


def evaluate_churn(
    data: ChurnData,
    bases: tuple[str, ...] = BASES,
    calibrations: tuple[str, ...] = ("none", "isotonic"),
    *,
    test_size: float = 0.25,
    seed: int = 0,
) -> list[ChurnScore]:
    """Fit every base and calibration on one stratified split and score each on the holdout."""
    _check_xy(data.x, data.y)
    unknown = [b for b in bases if b not in BASES] + [
        c for c in calibrations if c not in CALIBRATIONS
    ]
    if unknown or not bases or not calibrations:
        raise ValueError(f"unknown or missing base/calibration: {unknown}")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    x_tr, x_te, y_tr, y_te = train_test_split(
        data.x, data.y, test_size=test_size, stratify=data.y, random_state=seed
    )
    out = []
    for base in bases:
        for calibration in calibrations:
            start = time.perf_counter()
            model = fit_churn(x_tr, y_tr, base, calibration, seed=seed)
            out.append(score(model, x_te, y_te, time.perf_counter() - start))
    return out
