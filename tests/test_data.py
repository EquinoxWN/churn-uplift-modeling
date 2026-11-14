"""Loading and validation of the Telco and Hillstrom files, and the verified download."""

from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from churn_uplift_modeling import data as d
from churn_uplift_modeling.data import (
    HILLSTROM_COLUMNS,
    TELCO_COLUMNS,
    DataError,
    download,
    load_hillstrom,
    load_telco,
)

from .conftest import TELCO_ROW, hillstrom_rows, telco_rows, write_csv


def test_telco_is_one_hot_encoded_with_stable_columns(telco_csv: Path) -> None:
    data = load_telco(telco_csv)
    assert data.x.shape == (6, len(data.names))
    assert data.y.tolist() == [0, 1, 0, 1, 0, 1]
    row = dict(zip(data.names, data.x[0], strict=True))
    assert row["tenure"] == 12
    assert row["MonthlyCharges"] == pytest.approx(70.5)
    assert row["Partner"] == 1
    assert row["Dependents"] == 0
    assert row["InternetService=Fiber optic"] == 1
    assert row["InternetService=DSL"] == 0
    assert row["TechSupport=No internet service"] == 1
    # Every multi-valued category contributes exactly one 1 per row.
    contract = [v for k, v in row.items() if k.startswith("Contract=")]
    assert sorted(contract) == [0, 0, 1]
    assert data.names == d.telco_feature_names()


def test_blank_total_charges_is_zero_only_in_the_first_month(tmp_path: Path) -> None:
    rows = telco_rows(2)
    rows[0].update(tenure="0", TotalCharges=" ")
    data = load_telco(write_csv(tmp_path / "t.csv", TELCO_COLUMNS, rows))
    assert data.x[0, d.telco_feature_names().index("TotalCharges")] == 0.0
    rows[0].update(tenure="3")
    with pytest.raises(DataError, match="TotalCharges"):
        load_telco(write_csv(tmp_path / "t2.csv", TELCO_COLUMNS, rows))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Churn", "Maybe", "Churn"),
        ("Contract", "Weekly", "Contract"),
        ("gender", "", "gender"),
        ("tenure", "-1", "tenure"),
        ("tenure", "1.5", "tenure"),
        ("SeniorCitizen", "2", "SeniorCitizen"),
        ("MonthlyCharges", "abc", "MonthlyCharges"),
        ("MonthlyCharges", "nan", "MonthlyCharges"),
        ("TotalCharges", "-5", "TotalCharges"),
        ("customerID", "", "customerID"),
    ],
)
def test_malformed_telco_values_are_refused(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    rows = telco_rows(3)
    rows[1][field] = value
    with pytest.raises(DataError, match=message):
        load_telco(write_csv(tmp_path / "t.csv", TELCO_COLUMNS, rows))


def test_repeated_ids_missing_columns_and_empty_files_are_refused(tmp_path: Path) -> None:
    rows = telco_rows(3)
    rows[2]["customerID"] = rows[0]["customerID"]
    with pytest.raises(DataError, match="repeated"):
        load_telco(write_csv(tmp_path / "a.csv", TELCO_COLUMNS, rows))
    with pytest.raises(DataError, match="missing columns"):
        load_telco(write_csv(tmp_path / "b.csv", TELCO_COLUMNS[:-1], telco_rows(2)))
    with pytest.raises(DataError, match="no data rows"):
        load_telco(write_csv(tmp_path / "c.csv", TELCO_COLUMNS, []))
    with pytest.raises(DataError, match="missing file"):
        load_telco(tmp_path / "absent.csv")
    short = tmp_path / "d.csv"
    short.write_text(",".join(TELCO_COLUMNS) + "\n0001-A,Female\n", encoding="utf-8")
    with pytest.raises(DataError, match="number of fields"):
        load_telco(short)
    binary = tmp_path / "e.csv"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DataError, match="UTF-8"):
        load_telco(binary)


def test_oversized_files_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_csv(tmp_path / "t.csv", TELCO_COLUMNS, telco_rows(3))
    monkeypatch.setattr(d, "MAX_FILE_BYTES", path.stat().st_size - 1)
    with pytest.raises(DataError, match="larger than"):
        load_telco(path)


def test_hillstrom_arms_select_the_right_customers(hillstrom_csv: Path) -> None:
    womens = load_hillstrom(hillstrom_csv, "womens")
    mens = load_hillstrom(hillstrom_csv, "mens")
    both = load_hillstrom(hillstrom_csv, "any")
    assert womens.t.tolist() == [1, 1, 0, 0]
    assert mens.t.tolist() == [1, 1, 0, 0]
    assert both.t.tolist() == [1, 1, 1, 1, 0, 0]
    assert womens.y.tolist() == [1, 0, 1, 0]
    assert womens.x.shape == (4, len(d.HILLSTROM_FEATURES))
    row = dict(zip(womens.names, womens.x[1], strict=True))
    assert row["zip_code=Rural"] == 1
    assert row["channel=Web"] == 1
    assert row["channel=Phone"] == 0
    assert row["history"] == pytest.approx(142.44)
    assert row["recency"] == 10


def test_hillstrom_outcome_can_be_conversion(tmp_path: Path) -> None:
    rows = hillstrom_rows()
    rows[0]["conversion"] = "0"  # a visit without a purchase
    data = load_hillstrom(
        write_csv(tmp_path / "h.csv", HILLSTROM_COLUMNS, rows), "any", "conversion"
    )
    assert data.y.tolist() == [0, 0, 1, 0, 1, 0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("segment", "SMS", "segment"),
        ("recency", "13", "recency"),
        ("recency", "0", "recency"),
        ("history", "-1", "history"),
        ("mens", "yes", "mens"),
        ("zip_code", "Suburban", "zip_code"),
        ("channel", "Email", "channel"),
        ("visit", "2", "visit"),
        ("spend", "inf", "spend"),
    ],
)
def test_malformed_hillstrom_values_are_refused(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    rows = hillstrom_rows()
    rows[3][field] = value
    with pytest.raises(DataError, match=message):
        load_hillstrom(write_csv(tmp_path / "h.csv", HILLSTROM_COLUMNS, rows))


def test_hillstrom_needs_consistent_outcomes_and_both_arms(tmp_path: Path) -> None:
    rows = hillstrom_rows()
    rows[1]["conversion"] = "1"  # bought without visiting
    with pytest.raises(DataError, match="conversion without a visit"):
        load_hillstrom(write_csv(tmp_path / "a.csv", HILLSTROM_COLUMNS, rows))
    no_control = [r for r in hillstrom_rows() if r["segment"] != "No E-Mail"]
    with pytest.raises(DataError, match="both treated and control"):
        load_hillstrom(write_csv(tmp_path / "b.csv", HILLSTROM_COLUMNS, no_control))
    with pytest.raises(ValueError, match="arm"):
        load_hillstrom(tmp_path / "a.csv", "kids")
    with pytest.raises(ValueError, match="outcome"):
        load_hillstrom(tmp_path / "a.csv", "womens", "spend")


class FakeResponse(io.BytesIO):
    """A urlopen response serving fixed bytes."""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def opener_for(body: bytes, failures: int = 0) -> tuple[Any, list[float]]:
    """An opener that fails `failures` times, then serves body; also returns the timeouts seen."""
    timeouts: list[float] = []

    def opener(url: str, timeout: float) -> FakeResponse:
        timeouts.append(timeout)
        if len(timeouts) <= failures:
            raise urllib.error.URLError("connection reset")
        return FakeResponse(body)

    return opener, timeouts


BODY = b"customerID,Churn\n1,No\n"
DIGEST = hashlib.sha256(BODY).hexdigest()


def test_a_download_is_kept_only_when_the_checksum_matches(tmp_path: Path) -> None:
    opener, timeouts = opener_for(BODY)
    dest = tmp_path / "data" / "t.csv"
    assert download("https://example.test/t.csv", dest, DIGEST, opener=opener) == dest
    assert dest.read_bytes() == BODY
    assert timeouts == [60.0]
    bad, _ = opener_for(b"tampered")
    other = tmp_path / "data" / "u.csv"
    with pytest.raises(DataError, match="checksum mismatch"):
        download("https://example.test/u.csv", other, DIGEST, opener=bad)
    assert not other.exists()
    assert list(other.parent.glob("*.part")) == []


def test_a_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    dest = tmp_path / "t.csv"
    dest.write_bytes(BODY)
    opener, timeouts = opener_for(b"never fetched")
    download("https://example.test/t.csv", dest, DIGEST, opener=opener)
    assert timeouts == []


def test_only_https_is_allowed(tmp_path: Path) -> None:
    opener, timeouts = opener_for(BODY)
    with pytest.raises(DataError, match="non-HTTPS"):
        download("http://example.test/t.csv", tmp_path / "t.csv", DIGEST, opener=opener)
    assert timeouts == []


def test_the_size_cap_keeps_the_limit_and_refuses_more(tmp_path: Path) -> None:
    opener, _ = opener_for(BODY)
    download("https://example.test/t", tmp_path / "a", DIGEST, max_bytes=len(BODY), opener=opener)
    with pytest.raises(DataError, match="larger than"):
        download(
            "https://example.test/t", tmp_path / "b", DIGEST, max_bytes=len(BODY) - 1, opener=opener
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]


def test_transient_failures_are_retried_with_backoff(tmp_path: Path) -> None:
    opener, timeouts = opener_for(BODY, failures=2)
    sleeps: list[float] = []
    download("https://example.test/t", tmp_path / "t", DIGEST, opener=opener, sleep=sleeps.append)
    assert len(timeouts) == 3
    assert sleeps == [1.0, 2.0]


def test_retries_are_bounded_and_leave_no_partial_file(tmp_path: Path) -> None:
    opener, timeouts = opener_for(BODY, failures=10)
    sleeps: list[float] = []
    with pytest.raises(DataError, match="after 3 attempts"):
        download(
            "https://example.test/t", tmp_path / "t", DIGEST, opener=opener, sleep=sleeps.append
        )
    assert len(timeouts) == 3
    assert sleeps == [1.0, 2.0]
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="attempts"):
        download("https://example.test/t", tmp_path / "t", DIGEST, attempts=0, opener=opener)


def test_feature_values_are_plain_floats(telco_csv: Path) -> None:
    data = load_telco(telco_csv)
    assert data.x.dtype == np.float64
    assert data.y.dtype == np.int64
    assert np.isfinite(data.x).all()
    assert TELCO_ROW["customerID"] not in data.names
