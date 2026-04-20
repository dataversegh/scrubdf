"""Shared helper utilities used across scrubdf modules.

This module provides:
    - ``PipelineLog``: per-run log collector that also emits to Python's logging
    - ``ScrubError`` / ``ScrubTypeError`` / ``ScrubValueError``: typed exceptions
    - ``modified_z``: Modified Z-score computation
    - ``convert_series``: string-to-numeric Series conversion
    - ``validate_dataframe``: input guard with size warnings
    - ``is_string_dtype``: pandas 2/3 compatible string dtype check
"""

from __future__ import annotations

import datetime
import logging
import re
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
    Python's ``logging`` module under the ``scrubdf`` logger so they appear
    in structured logging setups (Cloud Logging, etc.).

    Examples
    --------
    >>> plog = PipelineLog()
    >>> plog.log("Dropped 5 duplicates")
    >>> len(plog)
    1
    """

    entries: List[str] = field(default_factory=list)

    def log(self, message: str, *, level: int = logging.INFO) -> str:
        """Record a timestamped log message.

        Parameters
        ----------
        message : str
            Human-readable description of what happened.
        level : int
            Python logging level (default ``logging.INFO``).

        Returns
        -------
        str
            The formatted, timestamped message.
        """
        timestamp = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.entries.append(timestamp)
        logger.log(level, message)
        return timestamp

    def warn(self, message: str) -> str:
        """Convenience: log at WARNING level."""
        return self.log(message, level=logging.WARNING)

    def error(self, message: str) -> str:
        """Convenience: log at ERROR level."""
        return self.log(message, level=logging.ERROR)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        # Always truthy — an empty log is still a valid log to write to.
        return True

    def __iter__(self):
        return iter(self.entries)


# ---------------------------------------------------------------------------
# Modified Z-score (used by outlier detection)
# ---------------------------------------------------------------------------

def modified_z(series: pd.Series) -> pd.Series | np.ndarray:
    """Compute modified Z-scores using the Median Absolute Deviation.

    Parameters
    ----------
    series : pd.Series
        Numeric series to score.

    Returns
    -------
    pd.Series or np.ndarray
        Z-scores.  If MAD is zero (constant column), returns an array
        of zeros so no values are flagged as outliers.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series))
    return 0.6745 * (series - median) / mad


# ---------------------------------------------------------------------------
# Dtype helpers (pandas 2/3 compatible)
# ---------------------------------------------------------------------------

def is_string_dtype(dtype) -> bool:
    """Check if a dtype is any kind of string (``object`` or ``StringDtype``).

    Works on both pandas 2 (where strings are ``object``) and pandas 3
    (where strings are ``StringDtype``).
    """
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


# ---------------------------------------------------------------------------
# Series type converter
# ---------------------------------------------------------------------------

def convert_series(series: pd.Series) -> pd.Series:
    """Try to convert a string Series to numeric.

    Strips commas and whitespace first, then checks if every non-null value
    matches a numeric pattern.  Returns the converted Series or the
    original if conversion isn't appropriate.

    Parameters
    ----------
    series : pd.Series
        Series with string dtype.

    Returns
    -------
    pd.Series
        Numeric Series if conversion succeeds, otherwise the original.
    """
    try:
        cleaned = series.astype("str").str.replace(",", "", regex=False).str.strip()
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
    """Guard: raise early if the input isn't a usable DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The candidate DataFrame.
    max_rows_warn : int
        Emit a ``UserWarning`` if row count exceeds this.
    max_rows_limit : int
        Raise ``ScrubValueError`` if row count exceeds this.

    Raises
    ------
    ScrubTypeError
        If *df* is not a pandas DataFrame.
    ScrubValueError
        If *df* is empty, has no columns, or exceeds *max_rows_limit*.
    """
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
            f"{max_rows_limit:,}-row safety limit. Split your data or "
            f"increase max_rows_limit if you're sure."
        )
    if len(df) > max_rows_warn:
        warnings.warn(
            f"DataFrame has {len(df):,} rows. Processing large datasets "
            f"may be slow and memory-intensive.",
            UserWarning,
            stacklevel=3,
        )
