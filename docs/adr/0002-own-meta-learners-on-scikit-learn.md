# ADR 0002: Own S/T/X meta-learners on scikit-learn instead of CausalML

- **Status:** Accepted (2026-11-14)

## Context

M1 needs uplift estimates from a randomized campaign. CausalML and scikit-uplift both provide S-, T-
and X-learners. They also bring a large dependency tree (CausalML adds compiled code, and pulls in
packages that the rest of the project does not use), and their behaviour is harder to test against
planted effects than code that is a few dozen lines long. The project also has to refuse
non-randomized data before fitting, which neither library does by default.

## Decision

Implement the three meta-learners in `uplift.py` on top of scikit-learn estimators (logistic or
ridge regression, or histogram gradient boosting). The X-learner uses the constant treated share
as its propensity, because M1 accepts only randomized data, and `check_randomized` gates every
uplift run (largest abs(SMD) must be <= 0.1).

## Consequences

- Two runtime dependencies (NumPy, scikit-learn); the learners are tested against known effects on
  synthetic data, and two hand mutations in them were only caught after tests were added for them.
- The X-learner is not usable on observational data without a propensity model; that is
  intentional for M1 and documented.
- Comparing with CausalML's implementations is left as the upstream contribution and as a check
  for M2.
- Uplift trees and causal forests are not available; they would need either more own code or the
  dependency this ADR avoids.
