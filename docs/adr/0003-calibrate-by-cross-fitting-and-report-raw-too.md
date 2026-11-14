# ADR 0003: Calibrate by cross-fitting, and always report the raw model next to it

- **Status:** Accepted (2026-11-14)

## Context

Calibrated churn probabilities are needed so expected losses can be computed. Calibration fitted on
the same data as the classifier learns the classifier's training-set confidence and is biased;
calibration on a separate split wastes data. Calibration also does not always help: a well-specified
logistic regression is already close to calibrated, and isotonic regression adds variance.

## Decision

Calibrate with `CalibratedClassifierCV` (5 folds, `ensemble=True`): each fold's calibrator is fitted
on predictions of a classifier that did not see those customers, and the final prediction averages
the fold models. Isotonic is the default; sigmoid (Platt) is selectable with `--calibration sigmoid`.
Every `cum churn` run fits each base model raw and calibrated on the same holdout and prints both.

## Consequences

- On the Telco holdout calibration lowered ECE for gradient boosting (0.0355 to 0.0232) and naive
  Bayes (0.2823 to 0.0373) but raised it for logistic regression (0.0133 to 0.0217); the table shows
  this instead of hiding it, and the choice per model is left to the user.
- Training takes about five times as long (five fold models instead of one).
- Probabilities from isotonic calibration are step-shaped: many customers share the same value,
  which matters when ranking (AUC moves by up to 0.003 on Telco).
