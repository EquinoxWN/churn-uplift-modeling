"""Uplift learners: recovery of planted effects, the randomization check and refused inputs."""

from __future__ import annotations

import numpy as np
import pytest

from churn_uplift_modeling.data import CampaignData
from churn_uplift_modeling.synthetic import SEGMENTS, SyntheticCampaign, make_campaign
from churn_uplift_modeling.uplift import (
    SLearner,
    TLearner,
    XLearner,
    check_campaign,
    check_randomized,
    evaluate_uplift,
    make_learner,
)

Pair = tuple[SyntheticCampaign, SyntheticCampaign]


@pytest.fixture(scope="module")
def campaign_train_test() -> Pair:
    """Independent training and test campaigns with known effects."""
    return make_campaign(16000, seed=11), make_campaign(6000, seed=12)


@pytest.mark.parametrize(
    ("name", "base"),
    [("t", "logistic"), ("x", "logistic"), ("t", "gbm"), ("x", "gbm"), ("s", "gbm")],
)
def test_learners_recover_the_planted_effects(
    campaign_train_test: Pair, name: str, base: str
) -> None:
    train, test = campaign_train_test
    d = train.data
    learner = make_learner(name, base).fit(d.x, d.t, d.y)
    pred = learner.predict(test.data.x)
    truth = test.true_uplift
    seg = test.segment
    assert abs(pred.mean() - truth.mean()) < 0.02
    assert np.corrcoef(pred, truth)[0, 1] > 0.7
    # Persuadables gain from the offer, 'do not disturb' customers lose.
    assert pred[seg == "persuadable"].mean() > 0.1
    assert pred[seg == "do_not_disturb"].mean() < 0.0


def test_a_linear_s_learner_cannot_see_who_responds(campaign_train_test: Pair) -> None:
    # Without interaction terms the treatment column only shifts every customer's logit equally.
    train, test = campaign_train_test
    d = train.data
    pred = SLearner("logistic").fit(d.x, d.t, d.y).predict(test.data.x)
    assert np.corrcoef(pred, test.true_uplift)[0, 1] < 0.3
    assert (pred > 0).all() or (pred < 0).all()


def test_no_effect_gives_predictions_near_zero() -> None:
    syn = make_campaign(12000, seed=13)
    d = syn.data
    rng = np.random.default_rng(0)
    shuffled_t = rng.permutation(d.t)  # treatment unrelated to outcome
    for learner in (TLearner("logistic"), XLearner("logistic")):
        pred = learner.fit(d.x, shuffled_t, d.y).predict(d.x)
        assert abs(pred.mean()) < 0.02
        assert np.abs(pred).max() < 0.15


def test_x_learner_uses_the_treated_share_as_propensity() -> None:
    syn = make_campaign(8000, seed=14, treated_share=0.2)
    learner = XLearner("logistic").fit(syn.data.x, syn.data.t, syn.data.y)
    assert learner.propensity == pytest.approx(syn.data.t.mean())
    assert 0.17 < learner.propensity < 0.23
    pred = learner.predict(syn.data.x)
    assert ((pred >= -1) & (pred <= 1)).all()


def test_evaluate_reports_observed_effects_and_agreement_with_truth() -> None:
    syn = make_campaign(10000, seed=15)
    scores = evaluate_uplift(syn.data, ("t", "x"), "logistic", true_uplift=syn.true_uplift, seed=0)
    assert [s.learner for s in scores] == ["t", "x"]
    for s in scores:
        assert s.observed.n_treated + s.observed.n_control == 3000
        assert s.top_observed.n_treated + s.top_observed.n_control == 900
        assert s.observed.low < s.mean_predicted < s.observed.high
        # Customers ranked highest really respond more in the holdout than the rest.
        assert s.top_observed.estimate > s.rest_observed.estimate + 0.1
        assert s.corr_true is not None
        assert s.corr_true > 0.6
        assert s.rmse_true is not None
        assert s.rmse_true < 0.2
    without_truth = evaluate_uplift(syn.data, ("t",), "logistic", seed=0)
    assert without_truth[0].corr_true is None
    assert without_truth[0].rmse_true is None


def test_the_randomization_check_flags_targeted_offers() -> None:
    syn = make_campaign(5000, seed=16)
    assert check_randomized(syn.data).balanced
    # Offer sent to customers with high x0 only: the arms differ before the offer.
    targeted = (syn.data.x[:, 0] > 0).astype(np.int64)
    skewed = CampaignData(syn.data.x, targeted, syn.data.y, syn.data.names)
    balance = check_randomized(skewed)
    assert not balance.balanced
    assert balance.worst_feature == "x0"
    assert balance.max_abs_smd > 1.0


def test_synthetic_segments_cover_all_four_groups() -> None:
    syn = make_campaign(5000, seed=17)
    counts = {s: int((syn.segment == s).sum()) for s in SEGMENTS}
    assert all(c > 200 for c in counts.values())
    assert (syn.true_uplift[syn.segment == "persuadable"] > 0).all()
    assert (syn.true_uplift[syn.segment == "do_not_disturb"] < 0).all()
    assert (syn.true_uplift[np.isin(syn.segment, ("sure_thing", "lost_cause"))] == 0).all()
    with pytest.raises(ValueError, match="treated_share"):
        make_campaign(100, treated_share=1.0)
    with pytest.raises(ValueError, match="at least 10"):
        make_campaign(5)


def test_bad_campaigns_and_calls_are_refused() -> None:
    syn = make_campaign(400, seed=18)
    x, t, y = syn.data.x, syn.data.t, syn.data.y
    with pytest.raises(ValueError, match="0 or 1"):
        check_campaign(x, t * 2, y)
    with pytest.raises(ValueError, match="one treatment flag"):
        check_campaign(x[:10], t, y)
    bad = x.copy()
    bad[3, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        check_campaign(bad, t, y)
    one_arm = np.ones_like(t)
    with pytest.raises(ValueError, match="treatment=0"):
        TLearner().fit(x, one_arm, y)
    with pytest.raises(RuntimeError, match="fit"):
        XLearner().predict(x)
    with pytest.raises(ValueError, match="learner"):
        make_learner("r")
    with pytest.raises(ValueError, match="base"):
        TLearner("svm")
    with pytest.raises(ValueError, match="unknown"):
        evaluate_uplift(syn.data, ("t", "dr"))
    with pytest.raises(ValueError, match="base"):
        evaluate_uplift(syn.data, ("t",), "svm")
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_uplift(syn.data, ("t",), top_share=0.0)
    with pytest.raises(ValueError, match="one value per customer"):
        evaluate_uplift(syn.data, ("t",), true_uplift=syn.true_uplift[:5])
    fitted = SLearner().fit(x, t, y)
    with pytest.raises(ValueError, match="finite 2-D"):
        fitted.predict(x[0])


class Constant:
    """A stand-in effect model that predicts one value for everyone."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value)


def test_x_learner_weights_the_control_effect_model_by_the_propensity() -> None:
    syn = make_campaign(4000, seed=19, treated_share=0.2)
    learner = XLearner("logistic").fit(syn.data.x, syn.data.t, syn.data.y)
    learner.tau0, learner.tau1, learner.propensity = Constant(0.1), Constant(0.3), 0.2
    # Kuenzel et al.: tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x).
    assert learner.predict(syn.data.x[:3]) == pytest.approx([0.26, 0.26, 0.26])


def test_a_moderate_imbalance_is_already_flagged() -> None:
    syn = make_campaign(20000, seed=20)
    x = syn.data.x.copy()
    x[:, 5] += 0.3 * syn.data.t  # treated customers differ by 0.3 standard deviations
    balance = check_randomized(CampaignData(x, syn.data.t, syn.data.y, syn.data.names))
    assert balance.worst_feature == "x5"
    assert 0.25 < balance.max_abs_smd < 0.35
    assert not balance.balanced
