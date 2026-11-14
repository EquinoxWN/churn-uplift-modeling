"""Command line: download the public data, score churn models and uplift learners on a holdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import churn, uplift
from .data import (
    ARMS,
    HILLSTROM_SHA256,
    HILLSTROM_URL,
    OUTCOMES,
    TELCO_SHA256,
    TELCO_URL,
    DataError,
    download,
    load_hillstrom,
    load_telco,
)
from .files import write_text
from .synthetic import make_campaign, make_churn

DATASETS = {
    "telco": (TELCO_URL, TELCO_SHA256, Path("data/telco.csv")),
    "hillstrom": (HILLSTROM_URL, HILLSTROM_SHA256, Path("data/hillstrom.csv")),
}
MAX_SYNTHETIC = 200_000


def _seed(text: str) -> int:
    """Parse a non-negative random seed."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("seed must be within [0, 2**32)")
    return value


def _size(text: str) -> int:
    """Parse a synthetic sample size."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if not 1000 <= value <= MAX_SYNTHETIC:
        raise argparse.ArgumentTypeError(f"size must be 1000 to {MAX_SYNTHETIC}")
    return value


def _names(allowed: Sequence[str]) -> Any:
    """Parser for a comma-separated subset of allowed names."""

    def parse(text: str) -> tuple[str, ...]:
        """Split and check the names."""
        items = tuple(dict.fromkeys(s.strip() for s in text.split(",") if s.strip()))
        if not items or any(i not in allowed for i in items):
            raise argparse.ArgumentTypeError(f"choose from {','.join(allowed)}: {text!r}")
        return items

    return parse


def _json_path(text: str) -> Path:
    """Accept only .json output paths."""
    path = Path(text)
    if path.suffix != ".json":
        raise argparse.ArgumentTypeError(f"output must end in .json: {text!r}")
    return path


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(prog="cum", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download a public dataset (SHA-256 checked)")
    dl.add_argument("dataset", choices=sorted(DATASETS))
    dl.add_argument("--dest", type=Path, help="target file (default data/<dataset>.csv)")

    ch = sub.add_parser("churn", help="fit churn models, raw and calibrated, and score a holdout")
    src = ch.add_mutually_exclusive_group()
    src.add_argument("--data", type=Path, default=DATASETS["telco"][2], help="Telco churn CSV")
    src.add_argument("--synthetic", action="store_true", help="generated data, known probabilities")
    ch.add_argument("--bases", type=_names(churn.BASES), default=churn.BASES)
    ch.add_argument(
        "--calibration", choices=[c for c in churn.CALIBRATIONS if c != "none"], default="isotonic"
    )

    up = sub.add_parser("uplift", help="fit uplift learners on randomized data and score a holdout")
    src = up.add_mutually_exclusive_group()
    src.add_argument("--data", type=Path, default=DATASETS["hillstrom"][2], help="Hillstrom CSV")
    src.add_argument("--synthetic", action="store_true", help="generated data, known effects")
    up.add_argument("--arm", choices=ARMS, default="womens", help="Hillstrom e-mail arm")
    up.add_argument("--outcome", choices=OUTCOMES, default="visit", help="Hillstrom outcome")
    up.add_argument("--learners", type=_names(uplift.LEARNERS), default=uplift.LEARNERS)
    up.add_argument("--base", choices=uplift.BASES, default="gbm")
    for sp in (ch, up):
        sp.add_argument("--size", type=_size, default=20_000, help="synthetic customers")
        sp.add_argument("--seed", type=_seed, default=0)
        sp.add_argument("--json", type=_json_path, help="also write the results as JSON")
    return p


def _write_json(path: Path | None, command: str, payload: dict[str, Any]) -> None:
    """Write results with a UTC timestamp, atomically."""
    if path is None:
        return
    doc = {"command": command, "created_utc": datetime.now(UTC).isoformat(), **payload}
    write_text(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _churn(args: argparse.Namespace) -> None:
    """Score each base model raw and calibrated."""
    if args.synthetic:
        data = make_churn(args.size, args.seed).data
        source = f"synthetic ({args.size:,} customers, seed {args.seed})"
    else:
        data = load_telco(args.data)
        source = args.data.name
    rate = float(data.y.mean())
    print(
        f"Data: {source}: {data.y.size:,} customers, {len(data.names)} features, churn rate {rate:.3f}"
    )
    print(
        "Holdout: stratified 25%; calibration fitted by 5-fold cross-fitting on the training part"
    )
    scores = churn.evaluate_churn(data, args.bases, ("none", args.calibration), seed=args.seed)
    print("| Base | Calibration | AUC | Brier | Log loss | ECE | Mean p | Churn rate | Fit s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in scores:
        print(
            f"| {s.base} | {s.calibration} | {s.auc:.4f} | {s.brier:.4f} | {s.log_loss:.4f} | "
            f"{s.ece:.4f} | {s.mean_predicted:.3f} | {s.observed_rate:.3f} | {s.fit_seconds:.1f} |"
        )
    _write_json(args.json, "churn", {"source": source, "scores": [asdict(s) for s in scores]})


def _effect(e: Any) -> str:
    """Format an effect with its 95% interval."""
    return f"{e.estimate:+.4f} [{e.low:+.4f}, {e.high:+.4f}]"


def _uplift(args: argparse.Namespace) -> None:
    """Score each learner on a randomized holdout."""
    truth = None
    if args.synthetic:
        syn = make_campaign(args.size, args.seed)
        data, truth = syn.data, syn.true_uplift
        source = f"synthetic ({args.size:,} customers, seed {args.seed})"
    else:
        data = load_hillstrom(args.data, args.arm, args.outcome)
        source = f"{args.data.name}, arm {args.arm} vs no e-mail, outcome {args.outcome}"
    balance = uplift.check_randomized(data)
    print(
        f"Data: {source}: {data.t.size:,} customers, treated share {balance.treated_share:.3f}, "
        f"largest |SMD| {balance.max_abs_smd:.4f} ({balance.worst_feature})"
    )
    if not balance.balanced:
        raise DataError(
            f"treatment does not look randomized (|SMD| {balance.max_abs_smd:.3f} > "
            f"{uplift.MAX_SMD} on {balance.worst_feature}); uplift needs randomized data"
        )
    scores = uplift.evaluate_uplift(
        data, args.learners, args.base, true_uplift=truth, seed=args.seed
    )
    print("Holdout: 30%, stratified by arm and outcome; effects are treated minus control rates")
    head = (
        "| Learner | Base | Mean predicted | Observed (95% CI) | Top 30% observed | Rest observed |"
    )
    extra = " Corr. with truth | RMSE vs truth |" if truth is not None else ""
    print(head + extra + " Fit s |")
    print("|---" * (6 + (2 if truth is not None else 0) + 1) + "|")
    for s in scores:
        row = (
            f"| {s.learner.upper()} | {s.base} | {s.mean_predicted:+.4f} | {_effect(s.observed)} | "
            f"{_effect(s.top_observed)} | {_effect(s.rest_observed)} |"
        )
        if truth is not None:
            row += f" {s.corr_true:.3f} | {s.rmse_true:.4f} |"
        print(row + f" {s.fit_seconds:.1f} |")
    _write_json(
        args.json,
        "uplift",
        {"source": source, "balance": asdict(balance), "scores": [asdict(s) for s in scores]},
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns 0 on success and 2 on bad input or data."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "download":
            url, digest, default = DATASETS[args.dataset]
            path = download(url, args.dest or default, digest)
            print(f"{args.dataset}: {path} (SHA-256 {digest})")
        elif args.command == "churn":
            _churn(args)
        else:
            _uplift(args)
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
