"""Churn models: calibration against known probabilities, holdout scoring and refused inputs."""

from __future__ import annotations

import numpy as np
import pytest

from churn_uplift_modeling.churn import evaluate_churn, fit_churn, make_classifier
from churn_uplift_modeling.data import ChurnData
from churn_uplift_modeling.metrics import ece
from churn_uplift_modeling.synthetic import make_churn


def test_calibrated_probabilities_track_the_true_probabilities() -> None:
    train, test = make_churn(8000, seed=1), make_churn(4000, seed=2)
    model = fit_churn(train.data.x, train.data.y, "gbm", "isotonic")
    p = model.predict_proba(test.data.x)
    assert np.mean(np.abs(p - test.true_p)) < 0.06
    assert ece(test.data.y, p) < 0.03
    assert ((p >= 0) & (p <= 1)).all()


def test_calibration_repairs_an_overconfident_model() -> None:
    # Naive Bayes treats copied features as independent evidence, so its scores are too extreme.
    def copies(x: np.ndarray) -> np.ndarray:
        return np.hstack([x] * 4)

    train, test = make_churn(6000, seed=3), make_churn(4000, seed=4)
    raw = fit_churn(copies(train.data.x), train.data.y, "naive_bayes", "none")
    cal = fit_churn(copies(train.data.x), train.data.y, "naive_bayes", "isotonic")
    raw_ece = ece(test.data.y, raw.predict_proba(copies(test.data.x)))
    cal_ece = ece(test.data.y, cal.predict_proba(copies(test.data.x)))
    assert raw_ece > 0.08
    assert cal_ece < raw_ece / 3
    sig = fit_churn(copies(train.data.x), train.data.y, "naive_bayes", "sigmoid")
    assert ece(test.data.y, sig.predict_proba(copies(test.data.x))) < raw_ece / 2


def test_risk_ranking_beats_chance_and_follows_the_planted_driver() -> None:
    syn = make_churn(6000, seed=5)
    model = fit_churn(syn.data.x, syn.data.y, "logistic", "isotonic")
    low, high = np.zeros((1, 6)), np.zeros((1, 6))
    high[0, 0], low[0, 0] = 2.0, -2.0  # x0 raises the true churn logit by 1.0 per unit
    assert model.predict_proba(high)[0] > model.predict_proba(low)[0] + 0.3


def test_evaluate_scores_every_base_raw_and_calibrated() -> None:
    syn = make_churn(3000, seed=6)
    scores = evaluate_churn(syn.data, ("logistic", "naive_bayes"), ("none", "isotonic"), seed=0)
    assert [(s.base, s.calibration) for s in scores] == [
        ("logistic", "none"),
        ("logistic", "isotonic"),
        ("naive_bayes", "none"),
        ("naive_bayes", "isotonic"),
    ]
    for s in scores:
        assert 0.7 < s.auc < 0.9
        assert sum(b.count for b in s.bins) == 750  # 25% holdout
        assert s.observed_rate == pytest.approx(scores[0].observed_rate)
        assert s.fit_seconds >= 0


def test_the_same_seed_gives_the_same_scores() -> None:
    syn = make_churn(2000, seed=7)
    a = evaluate_churn(syn.data, ("gbm",), ("isotonic",), seed=3)
    b = evaluate_churn(syn.data, ("gbm",), ("isotonic",), seed=3)
    assert (a[0].auc, a[0].brier, a[0].ece) == (b[0].auc, b[0].brier, b[0].ece)


def test_bad_inputs_are_refused() -> None:
    syn = make_churn(500, seed=8)
    x, y = syn.data.x, syn.data.y
    with pytest.raises(ValueError, match="base"):
        make_classifier("svm")
    with pytest.raises(ValueError, match="calibration"):
        fit_churn(x, y, "logistic", "platt")
    with pytest.raises(ValueError, match="one row per label"):
        fit_churn(x[:10], y, "logistic")
    with pytest.raises(ValueError, match="0 or 1"):
        fit_churn(x, y * 2, "logistic")
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_churn(bad, y, "logistic")
    few = y.copy()
    few[:] = 0
    few[:3] = 1
    with pytest.raises(ValueError, match="each class"):
        fit_churn(x, few, "logistic", "isotonic")
    with pytest.raises(ValueError, match="unknown"):
        evaluate_churn(syn.data, ("logistic", "svm"))
    with pytest.raises(ValueError, match="test_size"):
        evaluate_churn(syn.data, ("logistic",), test_size=1.0)
    model = fit_churn(x, y, "logistic", "none")
    with pytest.raises(ValueError, match="finite 2-D"):
        model.predict_proba(x[0])


def test_a_single_class_dataset_is_refused() -> None:
    data = ChurnData(x=np.zeros((20, 2)), y=np.zeros(20, dtype=np.int64), names=("a", "b"))
    with pytest.raises(ValueError, match=r"each class|least populated"):
        evaluate_churn(data, ("logistic",))
