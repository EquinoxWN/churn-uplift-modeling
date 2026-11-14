"""Verified downloads and validated loading of the Telco churn and Hillstrom campaign files."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# IBM's public sample of 7,043 telecom customers with a churn label (no sign-in needed).
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# No checksum is published upstream: SHA-256 of the file downloaded on 2026-11-14.
TELCO_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
# Kevin Hillstrom's MineThatData e-mail test (64,000 customers, randomized), HTTPS mirror.
# The original (http://www.minethatdata.com/...) is HTTP only; on 2026-11-14 both copies had the
# same SHA-256 after removing carriage returns, so the mirror holds the original rows.
HILLSTROM_URL = (
    "https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_SHA256 = "27bab8c5d3669f26ec08ebb50a0a78317542f29501156f2e2af6781fab4cd7e2"
MAX_DOWNLOAD_BYTES = 10_000_000
MAX_FILE_BYTES = 20_000_000

YES_NO = ("No", "Yes")
NO_INTERNET = ("No", "Yes", "No internet service")
# Telco categorical columns and every value allowed in each (one source for the encoding).
TELCO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": NO_INTERNET,
    "OnlineBackup": NO_INTERNET,
    "DeviceProtection": NO_INTERNET,
    "TechSupport": NO_INTERNET,
    "StreamingTV": NO_INTERNET,
    "StreamingMovies": NO_INTERNET,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
TELCO_NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
TELCO_COLUMNS = ("customerID", *TELCO_NUMERIC, *TELCO_CATEGORIES, "Churn")
MAX_TENURE = 1200
MAX_CHARGE = 1_000_000.0

HILLSTROM_COLUMNS = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
)
ZIP_CODES = ("Urban", "Surburban", "Rural")  # spelling as in the original file
CHANNELS = ("Phone", "Web", "Multichannel")
SEGMENTS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
ARMS = ("womens", "mens", "any")
OUTCOMES = ("visit", "conversion")


class DataError(ValueError):
    """Raised when a download or an input file is missing, tampered with or malformed."""


@dataclass(frozen=True)
class ChurnData:
    """Feature matrix, churn labels (1 = left) and feature names."""

    x: FloatArray
    y: IntArray
    names: tuple[str, ...]


@dataclass(frozen=True)
class CampaignData:
    """Feature matrix, treatment flags (1 = got the offer), binary outcomes and feature names."""

    x: FloatArray
    t: IntArray
    y: IntArray
    names: tuple[str, ...]


Opener = Callable[..., Any]


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    expected_sha256: str,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    timeout: float = 60.0,
    attempts: int = 3,
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download over HTTPS into dest, keeping it only if its SHA-256 matches."""
    if not url.startswith("https://"):
        raise DataError(f"refusing a non-HTTPS URL: {url}")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if dest.exists() and sha256(dest) == expected_sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Per-process partial file: two concurrent downloads never write into the same file.
    part = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    for attempt in range(1, attempts + 1):
        try:
            _fetch(url, part, max_bytes, timeout, opener)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise DataError(f"download failed after {attempts} attempts: {exc}") from exc
            sleep(2.0 ** (attempt - 1))
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    actual = sha256(part)
    if actual != expected_sha256:
        part.unlink(missing_ok=True)
        raise DataError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
    os.replace(part, dest)
    return dest


def _fetch(url: str, part: Path, max_bytes: int, timeout: float, opener: Opener) -> None:
    """Stream one response into part, enforcing the size cap."""
    total = 0
    with opener(url, timeout=timeout) as resp, part.open("wb") as out:
        while chunk := resp.read(1 << 16):
            total += len(chunk)
            if total > max_bytes:
                raise DataError(f"download larger than {max_bytes} bytes")
            out.write(chunk)


def _read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read a size-capped CSV and check that it has the expected columns and at least one row."""
    if not path.is_file():
        raise DataError(f"missing file: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{path.name} is larger than {MAX_FILE_BYTES} bytes")
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            missing = [c for c in columns if c not in header]
            if missing:
                raise DataError(f"{path.name}: missing columns {missing}")
            rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DataError(f"{path.name}: not a readable UTF-8 CSV ({exc})") from exc
    if not rows:
        raise DataError(f"{path.name}: no data rows")
    for i, row in enumerate(rows, start=2):
        if None in row or any(row[c] is None for c in columns):
            raise DataError(f"{path.name} line {i}: wrong number of fields")
    return rows


def _number(text: str, where: str, low: float, high: float) -> float:
    """Parse a finite number within [low, high]."""
    try:
        value = float(text)
    except ValueError as exc:
        raise DataError(f"{where}: not a number: {text!r}") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise DataError(f"{where}: {text!r} outside [{low}, {high}]")
    return value


def _whole(text: str, where: str, low: int, high: int) -> int:
    """Parse a whole number within [low, high]."""
    digits = text.strip().removeprefix("-")
    if not (digits.isascii() and digits.isdigit()):
        raise DataError(f"{where}: not a whole number: {text!r}")
    value = int(text)
    if not low <= value <= high:
        raise DataError(f"{where}: {value} outside [{low}, {high}]")
    return value


def _choice(text: str, where: str, allowed: Sequence[str]) -> str:
    """Check that a value is one of the allowed categories."""
    if text not in allowed:
        raise DataError(f"{where}: unexpected value {text!r}, allowed {list(allowed)}")
    return text


def telco_feature_names() -> tuple[str, ...]:
    """Column names of the Telco feature matrix, in order."""
    names = list(TELCO_NUMERIC)
    for col, values in TELCO_CATEGORIES.items():
        if values == YES_NO:
            names.append(col)
        else:
            names.extend(f"{col}={v}" for v in values)
    return tuple(names)


def load_telco(path: Path) -> ChurnData:
    """Load and validate the Telco churn CSV into one-hot features and 0/1 churn labels."""
    rows = _read_rows(path, TELCO_COLUMNS)
    seen: set[str] = set()
    features: list[list[float]] = []
    labels: list[int] = []
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        cid = row["customerID"].strip()
        if not cid:
            raise DataError(f"{where}: empty customerID")
        if cid in seen:
            raise DataError(f"{where}: repeated customerID {cid!r}")
        seen.add(cid)
        tenure = _whole(row["tenure"], f"{where} tenure", 0, MAX_TENURE)
        total_text = row["TotalCharges"].strip()
        # The public file leaves TotalCharges blank for customers in their first month only.
        if not total_text and tenure == 0:
            total = 0.0
        else:
            total = _number(total_text, f"{where} TotalCharges", 0.0, MAX_CHARGE)
        vec = [
            float(_whole(row["SeniorCitizen"], f"{where} SeniorCitizen", 0, 1)),
            float(tenure),
            _number(row["MonthlyCharges"], f"{where} MonthlyCharges", 0.0, MAX_CHARGE),
            total,
        ]
        for col, values in TELCO_CATEGORIES.items():
            value = _choice(row[col], f"{where} {col}", values)
            if values == YES_NO:
                vec.append(1.0 if value == "Yes" else 0.0)
            else:
                vec.extend(1.0 if value == v else 0.0 for v in values)
        features.append(vec)
        labels.append(1 if _choice(row["Churn"], f"{where} Churn", YES_NO) == "Yes" else 0)
    return ChurnData(
        x=np.asarray(features, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        names=telco_feature_names(),
    )


HILLSTROM_FEATURES = (
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    *(f"zip_code={z}" for z in ZIP_CODES),
    *(f"channel={c}" for c in CHANNELS),
)


def load_hillstrom(path: Path, arm: str = "womens", outcome: str = "visit") -> CampaignData:
    """Load the Hillstrom file: one e-mail arm (or any) against the no-e-mail control."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    rows = _read_rows(path, HILLSTROM_COLUMNS)
    wanted = {"womens": "Womens E-Mail", "mens": "Mens E-Mail"}.get(arm)
    features: list[list[float]] = []
    treated: list[int] = []
    outcomes: list[int] = []
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        segment = _choice(row["segment"], f"{where} segment", SEGMENTS)
        vec = [
            float(_whole(row["recency"], f"{where} recency", 1, 12)),
            _number(row["history"], f"{where} history", 0.0, MAX_CHARGE),
            float(_whole(row["mens"], f"{where} mens", 0, 1)),
            float(_whole(row["womens"], f"{where} womens", 0, 1)),
            float(_whole(row["newbie"], f"{where} newbie", 0, 1)),
        ]
        zip_code = _choice(row["zip_code"], f"{where} zip_code", ZIP_CODES)
        channel = _choice(row["channel"], f"{where} channel", CHANNELS)
        vec.extend(1.0 if zip_code == z else 0.0 for z in ZIP_CODES)
        vec.extend(1.0 if channel == c else 0.0 for c in CHANNELS)
        visit = _whole(row["visit"], f"{where} visit", 0, 1)
        conversion = _whole(row["conversion"], f"{where} conversion", 0, 1)
        _number(row["spend"], f"{where} spend", 0.0, MAX_CHARGE)
        if conversion and not visit:
            raise DataError(f"{where}: a conversion without a visit")
        if segment != "No E-Mail" and wanted is not None and segment != wanted:
            continue
        features.append(vec)
        treated.append(0 if segment == "No E-Mail" else 1)
        outcomes.append(visit if outcome == "visit" else conversion)
    t = np.asarray(treated, dtype=np.int64)
    if t.min() == t.max():
        raise DataError(f"{path.name}: needs both treated and control customers")
    return CampaignData(
        x=np.asarray(features, dtype=np.float64),
        t=t,
        y=np.asarray(outcomes, dtype=np.int64),
        names=HILLSTROM_FEATURES,
    )
