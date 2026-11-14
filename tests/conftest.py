"""Shared fixtures: small Telco- and Hillstrom-format CSV files written row by row."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from churn_uplift_modeling.data import HILLSTROM_COLUMNS, TELCO_COLUMNS

TELCO_ROW = {
    "customerID": "0001-A",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": "12",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No internet service",
    "StreamingTV": "No",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": "70.50",
    "TotalCharges": "846.00",
    "Churn": "Yes",
}

HILLSTROM_ROW = {
    "recency": "10",
    "history_segment": "2) $100 - $200",
    "history": "142.44",
    "mens": "1",
    "womens": "0",
    "zip_code": "Surburban",
    "newbie": "0",
    "channel": "Phone",
    "segment": "Womens E-Mail",
    "visit": "1",
    "conversion": "0",
    "spend": "0",
}


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> Path:
    """Write rows with the given header."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def telco_rows(n: int = 6) -> list[dict[str, str]]:
    """n valid Telco rows with alternating churn labels and distinct ids."""
    rows = []
    for i in range(n):
        row = dict(TELCO_ROW, customerID=f"{i:04d}-X", Churn="Yes" if i % 2 else "No")
        rows.append(row)
    return rows


def hillstrom_rows() -> list[dict[str, str]]:
    """Six valid rows: two per segment, one visit in each."""
    rows = []
    for segment in ("Womens E-Mail", "Mens E-Mail", "No E-Mail"):
        rows.append(dict(HILLSTROM_ROW, segment=segment, visit="1", conversion="1", spend="30"))
        rows.append(
            dict(HILLSTROM_ROW, segment=segment, visit="0", zip_code="Rural", channel="Web")
        )
    return rows


@pytest.fixture
def telco_csv(tmp_path: Path) -> Path:
    """A valid six-row Telco file."""
    return write_csv(tmp_path / "telco.csv", TELCO_COLUMNS, telco_rows())


@pytest.fixture
def hillstrom_csv(tmp_path: Path) -> Path:
    """A valid six-row Hillstrom file."""
    return write_csv(tmp_path / "hillstrom.csv", HILLSTROM_COLUMNS, hillstrom_rows())
