# RFC 0001: churn-uplift-modeling design

- **Status:** Accepted for M1 (M2 and M3 are open)
- **Author:** EquinoxWN
- **Created:** 2026-11-14

## Problem (Define)

Retention budgets are usually spent on the customers a churn model ranks as most likely to leave.
That ranking answers "who will leave?", not "whose decision does an offer change?". Some high-risk
customers leave whatever they are offered (lost causes), some stay anyway (sure things), and some
react badly to being contacted ("do not disturb"). Two separate estimates are needed:

1. a churn probability that can be read as a probability (calibrated), so expected losses and
   budgets can be computed from it, and
2. a per-customer treatment effect (uplift) estimated from a randomized campaign, so offers go to
   customers whose behaviour the offer changes.

Users: data scientists and analysts in retention, CRM and marketing teams who decide who receives
an offer, and engineers who need a tested reference for these two steps.

## Goals

M1 (this milestone):

- A churn model predicts who is likely to leave, with calibrated probabilities.
- Using randomized campaign data, uplift models estimate how much an offer changes each customer's
  behaviour.

Later milestones (not built yet):

- M2: customers are split into persuadables, sure things, lost causes and "do not disturb"; Qini
  curves and uplift-by-decile charts evaluate the models on a holdout set.
- M3: a campaign simulation compares targeting by churn risk with targeting by uplift under a
  fixed budget, and a write-up explains why the highest-risk customers are often the wrong ones
  to target.

### Critical-to-quality measures and targets (M1)

| Measure | Target | How it is measured |
|---|---|---|
| Churn ranking, Telco holdout | AUC >= 0.80 for the logistic and gradient-boosting models | `cum churn`, stratified 25% holdout |
| Churn calibration, Telco holdout | ECE (10 equal-width bins) <= 0.05 after calibration for every base model | same run |
| Calibration against known truth | mean absolute error to the true probability < 0.06 | `test_calibrated_probabilities_track_the_true_probabilities` |
| Uplift, average effect | mean predicted uplift inside the 95% interval of the observed holdout effect | `cum uplift`, 30% holdout |
| Uplift, ranking against known truth | correlation with the planted effect > 0.7 for T and X learners | `test_learners_recover_the_planted_effects` |
| Randomization gate | an assignment with any feature at abs(SMD) > 0.1 is refused | `test_a_moderate_imbalance_is_already_flagged` |

## Non-goals

- Running as a hosted production service or scoring customers online.
- Estimating effects from observational (non-randomized) data; M1 refuses it.
- Choosing who receives offers (segments, budgets, ROI); that is M2 and M3.
- GPU training or large models.

## Proposed design

`src/churn_uplift_modeling/`:

- **`data.py`:** HTTPS download with a timeout, a 10 MB cap, bounded retries and a pinned SHA-256;
  validated loaders for the IBM Telco churn file (7,043 customers, one-hot encoded from a fixed
  list of allowed categories) and the Hillstrom e-mail test (64,000 customers, one e-mail arm or
  both against the no-e-mail control, outcome visit or conversion).
- **`churn.py`:** three base models (standardized logistic regression, histogram gradient boosting,
  Gaussian naive Bayes as a known overconfident reference), each fitted raw and with cross-fitted
  calibration (`CalibratedClassifierCV`, 5 folds, isotonic or sigmoid). Every model is scored on the
  same stratified holdout: AUC, Brier score, log loss, ECE and the reliability bins.
- **`uplift.py`:** S-, T- and X-learners (Kuenzel et al., 2019) built on scikit-learn models with a
  logistic or gradient-boosting base, and `check_randomized`, which compares feature means between
  arms (standardized mean difference) before any learner is fitted.
- **`metrics.py`:** Brier, log loss, reliability bins, ECE, rank AUC, difference in means with a 95%
  interval, SMD; written out so each one is tested on worked examples.
- **`synthetic.py`:** customers with known churn probabilities, and a randomized campaign with four
  planted groups and a known effect per customer, so the tests can check against the truth.
- **`cli.py`:** `cum download`, `cum churn`, `cum uplift`, with an optional JSON report (UTC time
  stamp, atomic write).

Holdout evaluation in M1 for uplift is deliberately small: the observed difference in means on the
holdout, the same for the 30% of customers with the highest predicted uplift and for the rest.
Qini curves and deciles are M2.

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| CausalML (Uber) meta-learners | Pulls in a large dependency tree (and a compiled extension) for three learners that are a few dozen lines each on top of scikit-learn; own code is easier to test against planted effects. Kept as the reference to compare with and contribute to ([ADR 0002](../adr/0002-own-meta-learners-on-scikit-learn.md)). |
| scikit-uplift (sklift) | Same reasoning; its Hillstrom loader also downloads from a bucket that returned 404 on 2026-11-14. |
| Uplift trees / causal forests | Direct effect models; stronger on some data but harder to validate and slower. Candidate for M2 once Qini evaluation exists. |
| Platt (sigmoid) calibration as the default | Fewer parameters and smoother, but assumes a sigmoid-shaped distortion; isotonic is the default because the Telco holdout has enough customers, and sigmoid stays selectable ([ADR 0003](../adr/0003-calibrate-by-cross-fitting-and-report-raw-too.md)). |
| A churn label inside Hillstrom | Hillstrom has no churn label; using "no visit in two weeks" as churn would be a proxy with an 86% base rate. The public Telco file has a real churn label. |

## Measurement plan (Measure)

- `cum churn` on Telco (seed 0): AUC, Brier, log loss and ECE for each base model raw and
  calibrated on one stratified 25% holdout; the reliability bins are in the JSON report.
- `cum uplift` on Hillstrom (womens and mens arms, outcome visit, seed 0): treated share and
  largest SMD; per learner the mean predicted uplift against the observed holdout effect with its
  95% interval, and the observed effect in the top 30% against the rest.
- Synthetic data in tests and `--synthetic` runs: error against the true probability and
  correlation and RMSE against the true effect.
- Times come from `time.perf_counter`; results are in [docs/results/m1.md](../results/m1.md).

## Milestones

- **M1:** calibrated churn model; S/T/X uplift learners on randomized campaign data. Done.
- **M2:** segments; Qini curves and uplift by decile on a holdout.
- **M3:** budgeted campaign simulation (risk vs uplift targeting); write-up.

## Risks and open questions

- One holdout split per run: numbers move with the seed (ECE in particular, on 1,761 Telco holdout
  customers). M2 should add repeated splits or bootstrap intervals.
- Hillstrom effects are small (+4.5 points on visits) and the treatment-effect heterogeneity in it is
  weak, so learner differences on it are within noise; the synthetic data is where learners are
  separated.
- Both datasets are third-party copies pinned by hash on first download; their content is only as
  trustworthy as that first download (the Hillstrom mirror was compared with the original file).
- The Telco data is a fictional IBM sample; results do not describe a real company.
