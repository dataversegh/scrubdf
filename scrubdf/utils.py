"""Shared helper utilities used across scrubdf modules.

This module provides:
    - ``PipelineLog``: per-run log collector bridged to Python's logging
    - ``ScrubError`` hierarchy: typed exceptions
    - ``detect_id_columns``: heuristic ID column detection
    - ``modified_z``: Modified Z-score computation
    - ``convert_series``: string-to-numeric Series conversion
    - ``validate_dataframe``: input guard with size warnings
    - ``is_string_dtype``: pandas 2/3 compatible string dtype check
    - ``normalize_unicode``: encoding normalization for string columns
"""

from __future__ import annotations

import datetime
import logging
import re
import unicodedata
from unittest import result
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("scrubdf")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class ScrubError(Exception):
    """Base exception for all scrubdf errors."""


class ScrubTypeError(ScrubError, TypeError):
    """Raised when an argument has the wrong type."""


class ScrubValueError(ScrubError, ValueError):
    """Raised when an argument has an invalid value."""


class ScrubFileError(ScrubError):
    """Raised when a file cannot be read or parsed."""


# ---------------------------------------------------------------------------
# Pipeline logger
# ---------------------------------------------------------------------------

@dataclass
class PipelineLog:
    """Collects log messages for a single pipeline run.

    Each run gets its own ``PipelineLog`` instance — no globals, no file I/O,
    safe for concurrent API requests.  Messages are also forwarded to
    Python's ``logging`` module under the ``scrubdf`` logger.
    """

    entries: List[str] = field(default_factory=list)

    def log(self, message: str, *, level: int = logging.INFO) -> str:
        timestamp = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.entries.append(timestamp)
        logger.log(level, message)
        return timestamp

    def warn(self, message: str) -> str:
        return self.log(message, level=logging.WARNING)

    def error(self, message: str) -> str:
        return self.log(message, level=logging.ERROR)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return True

    def __iter__(self):
        return iter(self.entries)


# ---------------------------------------------------------------------------
# ID column detection
# ---------------------------------------------------------------------------

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|_id_|_key$|_code$|_index$|_no$|_num$|_number$|"
    r"^index$|^key$|^code$|^respondent|^participant|^record)",
    re.IGNORECASE,
)


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that are likely identifiers (not meaningful numerics).

    A column is flagged as an ID if:
    - Its name matches common ID patterns (id, key, code, index, etc.)
    - OR it's numeric/string with near-unique values (>95% unique)
      AND monotonically increasing/decreasing

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list of str
        Column names identified as likely IDs.
    """
    id_cols: list[str] = []
    n_rows = len(df)

    for col in df.columns:
        # Name-based detection (works even on empty DataFrames)
        if _ID_PATTERNS.search(str(col)):
            id_cols.append(col)
            continue

        # Behavior-based detection requires data
        if n_rows == 0:
            continue

        # Behavior-based detection for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            uniqueness_ratio = nunique / n_rows if n_rows > 0 else 0

            if uniqueness_ratio > 0.95:
                # Check if monotonic (typical of auto-increment IDs)
                non_null = df[col].dropna()
                if len(non_null) > 1:
                    is_monotonic = (
                        non_null.is_monotonic_increasing
                        or non_null.is_monotonic_decreasing
                    )
                    if is_monotonic:
                        id_cols.append(col)

    return id_cols


# ---------------------------------------------------------------------------
# Modified Z-score
# ---------------------------------------------------------------------------

def modified_z(series: pd.Series) -> pd.Series:
    """Compute modified Z-scores using the Median Absolute Deviation.

    Returns zeros for constant columns (MAD = 0). Preserves missing values as NaN.
    """
    s = pd.to_numeric(series, errors="coerce")

    if s.dropna().empty:
        return pd.Series(np.zeros(len(s), dtype="float64"), index=s.index)

    median = s.median()
    mad = np.median(np.abs(s.dropna() - median))

    if pd.isna(mad) or mad == 0:
        out = pd.Series(np.zeros(len(s), dtype="float64"), index=s.index)
        out[s.isna()] = np.nan
        return out

    result = pd.Series(0.6745 * (s - median) / mad, index=s.index, dtype="float64")
    return result


# ---------------------------------------------------------------------------
# Dtype helpers (pandas 2/3 compatible)
# ---------------------------------------------------------------------------

def is_string_dtype(dtype) -> bool:
    """Check if a dtype is any kind of string (``object`` or ``StringDtype``)."""
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


# ---------------------------------------------------------------------------
# Series type converter
# ---------------------------------------------------------------------------

def convert_series(series: pd.Series) -> pd.Series:
    """Try to convert a string Series to numeric.

    Strips commas and whitespace first, then checks if every non-null value
    matches a numeric pattern.
    """
    try:
        cleaned = series.astype("string").str.replace(r"[^0-9.-]", "", regex=True).str.strip()
        non_null = cleaned[cleaned.notna() & (cleaned != "<NA>") & (cleaned != "nan")]
        if len(non_null) == 0:
            return series
        is_numeric = non_null.apply(
            lambda x: bool(re.fullmatch(r"-?\d+(\.\d+)?", str(x)))
        ).all()
        if is_numeric:
            return pd.to_numeric(cleaned, errors="coerce")
        return series
    except Exception:
        return series


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize unicode text: curly quotes → straight, em dashes → hyphens, etc.

    Uses NFKC normalization and applies common replacements.
    """
    if not isinstance(text, str):
        return text
    # NFKC normalization handles most compatibility characters
    text = unicodedata.normalize("NFKC", text)
    # Additional common replacements
    replacements = {
        "\u2018": "'",   # left single curly quote
        "\u2019": "'",   # right single curly quote
        "\u201c": '"',   # left double curly quote
        "\u201d": '"',   # right double curly quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u00a0": " ",   # non-breaking space
        "\u2026": "...", # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

MAX_ROWS_WARNING: int = 500_000
MAX_ROWS_LIMIT: int = 10_000_000


def validate_dataframe(
    df: pd.DataFrame,
    *,
    max_rows_warn: int = MAX_ROWS_WARNING,
    max_rows_limit: int = MAX_ROWS_LIMIT,
) -> None:
    """Guard: raise early if the input isn't a usable DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ScrubTypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}"
        )
    if df.empty:
        raise ScrubValueError("DataFrame is empty — nothing to process")
    if len(df.columns) == 0:
        raise ScrubValueError("DataFrame has no columns")
    if len(df) > max_rows_limit:
        raise ScrubValueError(
            f"DataFrame has {len(df):,} rows, which exceeds the "
            f"{max_rows_limit:,}-row safety limit."
        )
    if len(df) > max_rows_warn:
        warnings.warn(
            f"DataFrame has {len(df):,} rows. Processing large datasets "
            f"may be slow and memory-intensive.",
            UserWarning,
            stacklevel=3,
        )