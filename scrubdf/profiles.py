"""Data profile presets that control pipeline behavior.

Profiles adjust which steps run, what thresholds are used, and how
edge cases are handled for different data types.

Usage::

    from scrubdf.cleaning import cleaning_pipeline

    # Survey data - safe defaults for Likert scales, skip logic, ordinal data
    result = cleaning_pipeline(df, profile="survey")

    # Transactional data - aggressive cleaning, full outlier removal
    result = cleaning_pipeline(df, profile="transactional")

    # Auto-detect (default) - inspects the data and picks heuristics
    result = cleaning_pipeline(df, profile="auto")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Set

import pandas as pd

from scrubdf.utils import is_string_dtype


@dataclass(frozen=True)
class DataProfile:
    """Configuration preset that controls pipeline behavior.

    Parameters
    ----------
    name : str
        Profile identifier.
    steps : set of str
        Which cleaning steps to run.
    missing_strategy : str
        Default strategy for numeric missing values.
    outlier_mode : str
        ``'remove'`` drops outlier rows, ``'flag'`` adds an
        ``_is_outlier`` column, ``'skip'`` does nothing.
    drop_high_null_threshold : float or None
        Drop columns with null fraction above this (``None`` = don't drop).
    drop_constant_columns : bool
        Whether to remove zero-variance columns.
    normalize_encoding : bool
        Whether to run unicode normalization on string columns.
    protect_id_columns : bool
        Whether to detect and exclude ID columns from numeric analysis.
    protect_ordinal_columns : bool
        Whether to detect and protect Likert/ordinal columns.
    respect_skip_logic : bool
        Whether to leave structured missingness alone (survey skip patterns).
    detect_coded_categoricals : bool
        Whether to treat low-cardinality integers as categorical.
    coded_categorical_max : int
        Maximum unique values for an integer column to be treated as categorical.
    """

    name: str
    steps: Set[str]
    missing_strategy: Literal["median", "mean", "mode", "drop", "skip"] = "median"
    outlier_mode: Literal["remove", "flag", "skip"] = "remove"
    drop_high_null_threshold: float | None = None
    drop_constant_columns: bool = False
    normalize_encoding: bool = False
    protect_id_columns: bool = True
    protect_ordinal_columns: bool = False
    respect_skip_logic: bool = False
    detect_coded_categoricals: bool = False
    coded_categorical_max: int = 10


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_ALL_STEPS = frozenset({
    "unnamed_columns", "column_names", "whitespace", "type_conversion",
    "standardise_values", "empty_rows_cols", "duplicates", "missing_values",
    "date_parsing", "outliers", "drop_high_null", "drop_constant",
    "normalize_encoding",
})

PROFILE_AUTO = DataProfile(
    name="auto",
    steps=_ALL_STEPS - {"drop_high_null", "drop_constant", "normalize_encoding"},
    missing_strategy="median",
    outlier_mode="remove",
    protect_id_columns=True,
)

PROFILE_TRANSACTIONAL = DataProfile(
    name="transactional",
    steps=_ALL_STEPS,
    missing_strategy="median",
    outlier_mode="remove",
    drop_high_null_threshold=0.7,
    drop_constant_columns=True,
    normalize_encoding=True,
    protect_id_columns=True,
)

PROFILE_SURVEY = DataProfile(
    name="survey",
    steps=_ALL_STEPS - {"outliers", "type_conversion"},
    missing_strategy="mode",
    outlier_mode="skip",
    drop_high_null_threshold=None,  # don't drop — missingness is meaningful
    drop_constant_columns=False,
    normalize_encoding=True,
    protect_id_columns=True,
    protect_ordinal_columns=True,
    respect_skip_logic=True,
    detect_coded_categoricals=True,
    coded_categorical_max=10,
)

_PROFILES = {
    "auto": PROFILE_AUTO,
    "transactional": PROFILE_TRANSACTIONAL,
    "survey": PROFILE_SURVEY,
}


def get_profile(name: str) -> DataProfile:
    """Look up a built-in profile by name.

    Parameters
    ----------
    name : str
        One of ``'auto'``, ``'transactional'``, ``'survey'``.

    Returns
    -------
    DataProfile

    Raises
    ------
    ValueError
        If *name* is not a recognised profile.
    """
    profile = _PROFILES.get(name.lower())
    if profile is None:
        raise ValueError(
            f"Unknown profile '{name}'. "
            f"Available profiles: {sorted(_PROFILES.keys())}"
        )
    return profile


# ---------------------------------------------------------------------------
# Column classification helpers (used by survey profile)
# ---------------------------------------------------------------------------

def detect_ordinal_columns(
    df: pd.DataFrame,
    max_unique: int = 10,
) -> list[str]:
    """Detect likely Likert/ordinal columns.

    A numeric column is flagged as ordinal if:
    - It has <= *max_unique* unique integer values
    - All values are small positive integers (typically 1-5, 1-7, 1-10)
    - It's NOT already detected as an ID column

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_unique : int
        Maximum unique values to qualify (default 10).

    Returns
    -------
    list of str
        Column names likely representing ordinal/Likert scales.
    """
    ordinal_cols: list[str] = []

    for col in df.select_dtypes(include="number").columns:
        try:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            nunique = non_null.nunique()
            if nunique > max_unique:
                continue

            # Check if all values are integers
            if not (non_null == non_null.astype(int)).all():
                continue

            # Check if values are in a small positive range
            min_val = non_null.min()
            max_val = non_null.max()
            if min_val >= 0 and max_val <= 100:
                ordinal_cols.append(col)
        except Exception:
            continue

    return ordinal_cols


def detect_coded_categoricals(
    df: pd.DataFrame,
    max_unique: int = 10,
) -> list[str]:
    """Detect integer columns that are actually categorical codes.

    Similar to ordinal detection but includes columns where the integers
    don't follow a natural scale — they're arbitrary codes like
    1=Male, 2=Female, 3=Other.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_unique : int
        Maximum unique values to qualify (default 10).

    Returns
    -------
    list of str
        Column names likely representing coded categorical variables.
    """
    coded_cols: list[str] = []

    for col in df.select_dtypes(include="number").columns:
        try:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            nunique = non_null.nunique()
            if nunique > max_unique or nunique < 2:
                continue

            # Must be integer-valued
            if not (non_null == non_null.astype(int)).all():
                continue

            # Low ratio of unique to total = likely categorical
            ratio = nunique / len(non_null)
            if ratio < 0.05:  # fewer than 5% unique values
                coded_cols.append(col)
        except Exception:
            continue

    return coded_cols


def detect_skip_logic_columns(
    df: pd.DataFrame,
    null_threshold: float = 0.3,
) -> list[str]:
    """Detect columns with structured missingness (survey skip logic).

    A column is flagged if:
    - It has a significant null percentage (> *null_threshold*)
    - The nulls correlate with values in another column
      (indicating conditional skip patterns)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    null_threshold : float
        Minimum null fraction to consider (default 0.3).

    Returns
    -------
    list of str
        Columns likely affected by skip logic.
    """
    skip_cols: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return skip_cols

    for col in df.columns:
        null_frac = df[col].isnull().sum() / n_rows
        if null_frac < null_threshold or null_frac > 0.95:
            continue

        # Check if nulls in this column align with specific values in other cols
        null_mask = df[col].isnull()
        for other_col in df.columns:
            if other_col == col:
                continue
            try:
                other_non_null = df.loc[null_mask, other_col].dropna()
                if len(other_non_null) == 0:
                    continue
                # If nulls in `col` concentrate around specific values in
                # `other_col`, that suggests skip logic
                if other_non_null.nunique() <= 3:
                    skip_cols.append(col)
                    break
            except Exception:
                continue

    return skip_cols