"""Command line: tables, JSON output, argument checks, exit codes and the pinned downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import churn_uplift_modeling
from churn_uplift_modeling import cli
from churn_uplift_modeling.data import (
    HILLSTROM_COLUMNS,
    HILLSTROM_SHA256,
    HILLSTROM_URL,
    TELCO_COLUMNS,
)

from .conftest import hillstrom_rows, telco_rows, write_csv


def test_churn_prints_the_table_and_writes_utc_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "churn.json"
    code = cli.main(
        ["churn", "--synthetic", "--size", "3000", "--bases", "logistic,gbm", "--json", str(out)]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "3,000 customers" in text
    assert text.count("| logistic |") == 2
    assert text.count("| gbm |") == 2
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["command"] == "churn"
    assert doc["created_utc"].endswith("+00:00")
    assert [(s["base"], s["calibration"]) for s in doc["scores"]] == [
        ("logistic", "none"),
        ("logistic", "isotonic"),
        ("gbm", "none"),
        ("gbm", "isotonic"),
    ]


def test_uplift_prints_truth_columns_for_synthetic_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "uplift.json"
    args = ["uplift", "--synthetic", "--size", "6000", "--learners", "t,x", "--base", "logistic"]
    assert cli.main([*args, "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Corr. with truth" in text
    assert "| T | logistic |" in text
    assert "| X | logistic |" in text
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["balance"]["balanced"] is True
    assert len(doc["scores"]) == 2
    assert doc["scores"][0]["corr_true"] > 0.5


def test_real_format_files_run_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [dict(r, customerID=f"{i:05d}") for i, r in enumerate(telco_rows(2) * 30)]
    for i, r in enumerate(rows):
        r["tenure"] = str(i % 40 + 1)
    telco = write_csv(tmp_path / "telco.csv", TELCO_COLUMNS, rows)
    assert cli.main(["churn", "--data", str(telco), "--bases", "logistic"]) == 0
    assert "telco.csv: 60 customers, 41 features" in capsys.readouterr().out
    hill = write_csv(tmp_path / "h.csv", HILLSTROM_COLUMNS, hillstrom_rows() * 20)
    assert (
        cli.main(
            ["uplift", "--data", str(hill), "--arm", "any", "--learners", "t", "--base", "logistic"]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "arm any vs no e-mail, outcome visit: 120 customers" in text
    assert "Corr. with truth" not in text


@pytest.mark.parametrize(
    "argv",
    [
        ["churn", "--bases", "svm"],
        ["churn", "--bases", ""],
        ["churn", "--size", "10"],
        ["churn", "--seed", "-1"],
        ["churn", "--json", "out.txt"],
        ["uplift", "--learners", "t,dr"],
        ["uplift", "--arm", "kids"],
        ["uplift", "--synthetic", "--data", "x.csv"],
        ["download", "kaggle"],
    ],
)
def test_bad_arguments_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    assert info.value.code == 2


def test_bad_data_exits_2_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["churn", "--data", str(tmp_path / "absent.csv")]) == 2
    assert "missing file" in capsys.readouterr().err
    rows = telco_rows(4)
    rows[0]["Churn"] = "Perhaps"
    bad = write_csv(tmp_path / "bad.csv", TELCO_COLUMNS, rows)
    assert cli.main(["churn", "--data", str(bad)]) == 2
    assert "Churn" in capsys.readouterr().err


def test_targeted_offers_are_refused_before_any_model_is_fitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every e-mailed customer is a newbie and no control customer is: not a randomized test.
    rows = [
        dict(r, newbie="0" if r["segment"] == "No E-Mail" else "1") for r in hillstrom_rows() * 10
    ]
    path = write_csv(tmp_path / "h.csv", HILLSTROM_COLUMNS, rows)
    assert cli.main(["uplift", "--data", str(path)]) == 2
    err = capsys.readouterr().err
    assert "does not look randomized" in err
    assert "newbie" in err


def test_download_uses_the_pinned_url_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path, str]] = []

    def fake(url: str, dest: Path, digest: str, **_: Any) -> Path:
        calls.append((url, dest, digest))
        return dest

    monkeypatch.setattr(cli, "download", fake)
    dest = tmp_path / "h.csv"
    assert cli.main(["download", "hillstrom", "--dest", str(dest)]) == 0
    assert calls == [(HILLSTROM_URL, dest, HILLSTROM_SHA256)]
    assert cli.main(["download", "telco"]) == 0
    assert calls[1][1] == Path("data/telco.csv")
    assert "SHA-256" in capsys.readouterr().out


def test_a_failed_write_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "churn.json"
    out.write_text("old", encoding="utf-8")

    def broken(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    code = cli.main(
        ["churn", "--synthetic", "--size", "1000", "--bases", "logistic", "--json", str(out)]
    )
    assert code == 2
    assert out.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["churn.json"]


def test_version() -> None:
    assert churn_uplift_modeling.__version__ == "0.1.0"
