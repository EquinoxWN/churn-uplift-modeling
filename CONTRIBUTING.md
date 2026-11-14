# Contributing

Thanks for helping. This is a Python project; issues and pull requests are welcome.

## Set up and check your change

Needs Python 3.11+ (NumPy and scikit-learn, CPU only). Caches, virtual environments and downloaded data stay in git-ignored folders inside the repo.

```bash
make setup   # install dependencies
make lint    # ruff, ruff format, mypy --strict
make test    # full test suite on hand-written files and synthetic data
make audit   # known-vulnerability check (as in CI)
make run     # download Telco and Hillstrom (checksum-verified) and score the models
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- Add or update a test for every behaviour change, and update `docs/results/` when numbers change.
- A new churn model must be scored raw and calibrated through `evaluate_churn`, on the same holdout.
- A new uplift learner must recover the planted effects in `make_campaign` (extend `test_learners_recover_the_planted_effects`).
- Uplift learners are only fitted on randomized data; keep the `check_randomized` gate in front of them.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
