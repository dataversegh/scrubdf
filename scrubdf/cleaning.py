"""DataFrame cleaning functions and the main cleaning pipeline.

Every function takes a DataFrame (and optional config) and returns
a cleaned DataFrame.  No Streamlit imports, no global state, no side effects
beyond the returned data.

All individual cleaning functions are safe to call standalone — they catch
per-column errors, log them, and continue processing remaining columns.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Set

import numpy as np
import pandas as pd

from scrubdf.utils import (
    PipelineLog,
    ScrubValueError,
    convert_series,
    is_string_dtype,
    modified_z,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def clean_unnamed_columns(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Drop all-null unnamed/blank columns; rename non-null ones.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance for this pipeline run.

    Returns
    -------
    pd.DataFrame
        DataFrame with unnamed columns handled.
    """
    renamed_cols: dict[str, str] = {}
    dropped_cols: list[str] = []
    counter = 1

    for col in df.columns.tolist():
        try:
            if "Unnamed" in str(col) or str(col).strip() == "":
                if df[col].isnull().all():
                    df = df.drop(columns=col)
                    dropped_cols.append(str(col))
                else:
                    new_col_name = f"renamed_{counter}"
                    while new_col_name in df.columns:
                        counter += 1
                        new_col_name = f"renamed_{counter}"
                    df = df.rename(columns={col: new_col_name})
                    renamed_cols[str(col)] = new_col_name
                    counter += 1
        except Exception as e:
            if plog:
                plog.error(f"Error processing unnamed column '{col}': {e}")

    if plog:
        if dropped_cols:
            plog.log(f"Dropped all-null unnamed columns: {dropped_cols}")
        if renamed_cols:
            plog.log(f"Renamed unnamed columns: {renamed_cols}")

    return df


def clean_col_name(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Standardise column names: lowercase, strip, replace spaces with underscores.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised column names.
    """
    original_columns = df.columns.tolist()
    df.columns = [
        str(col).strip().lower().replace(" ", "_") for col in df.columns
    ]
    if plog and df.columns.tolist() != original_columns:
        plog.log("Standardised column names")
    return df


def strip_whitespace(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string cells.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with whitespace stripped from string values.
    """
    try:
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        if plog:
            plog.log("Stripped whitespaces from string entries")
    except Exception as e:
        if plog:
            plog.error(f"Error stripping whitespace: {e}")
    return df


def convert_column_type(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Attempt to convert string columns to numeric where possible.

    Processes each column independently — a failure in one column
    does not affect others.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with eligible columns converted to numeric dtypes.
    """
    for col in df.columns:
        try:
            if is_string_dtype(df[col].dtype):
                before_dtype = df[col].dtype
                df[col] = convert_series(df[col])
                after_dtype = df[col].dtype
                if before_dtype != after_dtype and plog:
                    plog.log(f"Converted '{col}' from {before_dtype} to {after_dtype}")
        except Exception as e:
            if plog:
                plog.error(f"Error converting column '{col}': {e}")
    return df


def standardise_values(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Normalise common value variants in gender/sex columns.

    This is domain-specific to survey/demographic data.  Only applies
    to columns whose name contains 'gender' or 'sex'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised gender/sex values.
    """
    gender_map = {
        "m": "male", "boy": "male", "man": "male",
        "f": "female", "girl": "female", "woman": "female",
        "": "unknown", "n/a": "unknown", "na": "unknown",
    }
    for col in df.columns:
        try:
            if "gender" in col.lower() or "sex" in col.lower():
                if is_string_dtype(df[col].dtype):
                    df[col] = df[col].str.strip().str.lower().replace(gender_map)
                    if plog:
                        plog.log(f"Standardised values in '{col}'")
        except Exception as e:
            if plog:
                plog.error(f"Error standardising column '{col}': {e}")
    return df


def remove_duplicates(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove duplicate columns and duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (cleaned_df, list_of_duplicate_column_names)
    """
    # Duplicate columns
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        df = df.loc[:, ~df.columns.duplicated()]
        if plog:
            plog.log(f"Dropped duplicate columns: {dup_cols}")

    # Duplicate rows
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    after_rows = df.shape[0]
    if plog and before_rows != after_rows:
        plog.log(f"Dropped {before_rows - after_rows} duplicate rows")

    return df, dup_cols


def drop_fully_empty(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Drop columns and rows that are entirely null.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with fully-empty rows and columns removed.
    """
    empty_cols = df.columns[df.isnull().all()].tolist()
    empty_rows = df.index[df.isnull().all(axis=1)].tolist()

    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all")

    if plog:
        if empty_cols:
            plog.log(f"Dropped empty columns: {empty_cols}")
        if empty_rows:
            plog.log(f"Dropped {len(empty_rows)} empty rows")

    return df


_VALID_MISSING_STRATEGIES = {"median", "mean", "mode", "drop", "skip"}


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal["median", "mean", "mode", "drop", "skip"] = "median",
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Handle missing values in numeric and text columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str
        How to handle numeric nulls:

        - ``'median'``: fill with column median (default)
        - ``'mean'``: fill with column mean
        - ``'mode'``: fill with column mode
        - ``'drop'``: drop rows with any missing numeric values
        - ``'skip'``: leave as-is

        Text columns are always filled with ``'Unknown'`` (except on ``'skip'``).
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.

    Raises
    ------
    ScrubValueError
        If *strategy* is not one of the valid options.
    """
    if strategy not in _VALID_MISSING_STRATEGIES:
        raise ScrubValueError(
            f"Invalid missing-value strategy '{strategy}'. "
            f"Choose from: {sorted(_VALID_MISSING_STRATEGIES)}"
        )

    if strategy == "skip":
        if plog:
            plog.log("Skipped missing value treatment")
        return df

    if strategy == "drop":
        numeric_cols = df.select_dtypes(include="number").columns
        before = df.shape[0]
        df = df.dropna(subset=numeric_cols)
        if plog:
            plog.log(f"Dropped {before - df.shape[0]} rows with missing numeric values")
    else:
        for col in df.select_dtypes(include="number").columns:
            try:
                num_missing = df[col].isnull().sum()
                if num_missing > 0:
                    if strategy == "median":
                        fill_val = df[col].median()
                    elif strategy == "mean":
                        fill_val = df[col].mean()
                    elif strategy == "mode":
                        mode_vals = df[col].mode()
                        fill_val = mode_vals.iloc[0] if not mode_vals.empty else 0
                    else:
                        continue
                    df[col] = df[col].fillna(fill_val)
                    if plog:
                        plog.log(
                            f"Filled {num_missing} missing in '{col}' "
                            f"with {strategy}: {fill_val}"
                        )
            except Exception as e:
                if plog:
                    plog.error(f"Error filling missing values in '{col}': {e}")

    # Text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            text_missing = df[col].isnull().sum()
            if text_missing > 0:
                df[col] = df[col].str.strip()
                df[col] = df[col].fillna("Unknown")
                if plog:
                    plog.log(
                        f"Filled {text_missing} missing in '{col}' with 'Unknown'"
                    )
        except Exception as e:
            if plog:
                plog.error(f"Error filling text missing values in '{col}': {e}")

    return df


def parse_date_columns(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Auto-detect and parse date/timestamp columns.

    Only attempts conversion on columns whose name contains
    ``'date'`` or ``'timestamp'`` (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with date columns parsed.
    """
    for col in df.columns:
        if "date" in col.lower() or "timestamp" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if plog:
                    plog.log(f"Parsed '{col}' as datetime")
            except Exception as e:
                if plog:
                    plog.error(f"Could not parse '{col}' as datetime: {e}")
    return df


_VALID_OUTLIER_METHODS = {"IQR", "Z-score", "Isolation Forest"}


def detect_and_remove_outliers(
    df: pd.DataFrame,
    method: Literal["IQR", "Z-score", "Isolation Forest"] | None = None,
    plog: PipelineLog | None = None,
) -> tuple[pd.DataFrame, str, dict[str, int], dict[str, int]]:
    """Detect and remove outliers from numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str or None
        Detection method: ``'IQR'``, ``'Z-score'``, or ``'Isolation Forest'``.
        If ``None``, auto-selects based on data shape and skewness.
        Isolation Forest requires ``scikit-learn``
        (install with ``pip install scrubdf[ml]``).
    plog : PipelineLog, optional
        Logger instance.

    Returns
    -------
    tuple
        ``(cleaned_df, method_used, full_report, nonzero_report)``

    Raises
    ------
    ScrubValueError
        If *method* is not a recognised option.
    ImportError
        If Isolation Forest is requested but scikit-learn is not installed.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_report: dict[str, int] = {}

    if not numeric_cols:
        if plog:
            plog.warn("No numeric columns found — skipping outlier detection")
        return df, "skipped", {}, {}

    # Auto-select method
    if method is None:
        skewness = df[numeric_cols].skew().abs().mean()
        n_rows = df.shape[0]
        if n_rows < 1000:
            method = "Z-score" if skewness > 1 else "IQR"
        else:
            method = "Isolation Forest"
    elif method not in _VALID_OUTLIER_METHODS:
        raise ScrubValueError(
            f"Invalid outlier method '{method}'. "
            f"Choose from: {sorted(_VALID_OUTLIER_METHODS)}"
        )

    cleaned_df = df.copy()

    if method == "IQR":
        for col in numeric_cols:
            try:
                q1 = cleaned_df[col].quantile(0.25)
                q3 = cleaned_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = (cleaned_df[col] > upper) | (cleaned_df[col] < lower)
                outlier_report[col] = int(outliers.sum())
                cleaned_df = cleaned_df[~outliers]
            except Exception as e:
                if plog:
                    plog.error(f"IQR failed on column '{col}': {e}")

    elif method == "Z-score":
        for col in numeric_cols:
            try:
                mz = modified_z(cleaned_df[col])
                outliers = np.abs(mz) > 3.5
                outlier_report[col] = int(outliers.sum())
                cleaned_df = cleaned_df[~outliers]
            except Exception as e:
                if plog:
                    plog.error(f"Z-score failed on column '{col}': {e}")

    elif method == "Isolation Forest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "Isolation Forest requires scikit-learn. "
                "Install it with: pip install scrubdf[ml]"
            )
        try:
            iso = IsolationForest(contamination="auto", random_state=42)
            preds = iso.fit_predict(cleaned_df[numeric_cols])
            mask = preds == 1
            outlier_report["Total"] = int((~mask).sum())
            cleaned_df = cleaned_df[mask]
        except Exception as e:
            if plog:
                plog.error(f"Isolation Forest failed: {e}")
            outlier_report["Error"] = 0

    # Logging
    if plog:
        removed_count = df.shape[0] - cleaned_df.shape[0]
        plog.log(f"Outlier detection using {method}: {removed_count} rows removed")
        affected = [
            k for k, v in outlier_report.items()
            if isinstance(v, int) and v > 0
        ]
        if affected:
            plog.log(f"Outliers detected in columns: {affected}")

    outlier_report_kpi = {
        k: v for k, v in outlier_report.items()
        if isinstance(v, int) and v != 0
    }

    return cleaned_df, method, outlier_report, outlier_report_kpi


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

ALL_STEPS = frozenset({
    "unnamed_columns",
    "column_names",
    "whitespace",
    "type_conversion",
    "standardise_values",
    "empty_rows_cols",
    "duplicates",
    "missing_values",
    "date_parsing",
    "outliers",
})


def cleaning_pipeline(
    df: pd.DataFrame,
    *,
    steps: Set[str] | None = None,
    missing_strategy: Literal["median", "mean", "mode", "drop", "skip"] = "median",
    outlier_method: Literal["IQR", "Z-score", "Isolation Forest"] | None = None,
) -> Dict[str, Any]:
    """Run the full cleaning pipeline and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input DataFrame.  A copy is made internally — the
        original is never modified.
    steps : set of str, optional
        Which cleaning steps to run.  Defaults to all steps.
        Valid step names: see ``ALL_STEPS``.
    missing_strategy : str
        Strategy for handling missing values.
    outlier_method : str or None
        Outlier detection method.  ``None`` = auto-select.

    Returns
    -------
    dict
        Structured report with keys:

        - **cleaned_df** (*DataFrame*) — the cleaned data
        - **original_shape** (*tuple*) — ``(rows, cols)`` before cleaning
        - **cleaned_shape** (*tuple*) — ``(rows, cols)`` after cleaning
        - **nulls_handled** (*int*) — null values filled or dropped
        - **duplicates_removed** (*int*) — duplicate rows removed
        - **type_fixes** (*int*) — columns whose dtype changed
        - **columns_dropped** (*list*) — columns removed
        - **outliers_detected** (*dict*) — ``{column: count}``
        - **outlier_method** (*str*) — method used
        - **skew_report** (*DataFrame*) — skewness per column
        - **high_skew** (*list*) — highly skewed column descriptions
        - **skew_kpi** (*dict*) — top skewed columns
        - **logs** (*list*) — timestamped log messages

    Raises
    ------
    ScrubTypeError
        If *df* is not a DataFrame.
    ScrubValueError
        If *df* is empty, exceeds the row limit, or *steps* contains
        invalid step names.
    """
    validate_dataframe(df)
    df = df.copy()  # never mutate the caller's DataFrame

    # Validate step names
    active_steps = steps if steps is not None else ALL_STEPS
    invalid_steps = set(active_steps) - ALL_STEPS
    if invalid_steps:
        raise ScrubValueError(
            f"Invalid step name(s): {invalid_steps}. "
            f"Valid steps: {sorted(ALL_STEPS)}"
        )

    plog = PipelineLog()
    original_shape = df.shape
    original_nulls = int(df.isnull().sum().sum())
    original_cols = set(df.columns)
    dup_cols: list[str] = []

    plog.log(
        f"Pipeline started: {original_shape[0]:,} rows × "
        f"{original_shape[1]} columns, {original_nulls:,} null values"
    )

    # --- Prep & formatting ---
    if "unnamed_columns" in active_steps:
        df = clean_unnamed_columns(df, plog)
    if "column_names" in active_steps:
        df = clean_col_name(df, plog)
    if "whitespace" in active_steps:
        df = strip_whitespace(df, plog)
    if "type_conversion" in active_steps:
        before_dtypes = df.dtypes.to_dict()
        df = convert_column_type(df, plog)
        type_fixes = sum(
            1 for col in df.columns
            if col in before_dtypes and df[col].dtype != before_dtypes[col]
        )
    else:
        type_fixes = 0
    if "standardise_values" in active_steps:
        df = standardise_values(df, plog)

    # --- Structural cleaning ---
    if "empty_rows_cols" in active_steps:
        df = drop_fully_empty(df, plog)
    if "duplicates" in active_steps:
        before_dup = df.shape[0]
        df, dup_cols = remove_duplicates(df, plog)
        duplicates_removed = before_dup - df.shape[0]
    else:
        duplicates_removed = 0
    if "missing_values" in active_steps:
        df = handle_missing_values(df, strategy=missing_strategy, plog=plog)
    if "date_parsing" in active_steps:
        df = parse_date_columns(df, plog)

    # --- Quality checks ---
    if "outliers" in active_steps:
        df, method_used, outlier_report, outlier_report_kpi = (
            detect_and_remove_outliers(df, method=outlier_method, plog=plog)
        )
    else:
        method_used = "skipped"
        outlier_report = {}
        outlier_report_kpi = {}

    # --- Post-clean analysis ---
    remaining_nulls = int(df.isnull().sum().sum())
    nulls_handled = original_nulls - remaining_nulls
    columns_dropped = sorted(original_cols - set(df.columns))

    from scrubdf.eda import skewness_check
    skew_report_df, high_skew, skew_kpi = skewness_check(df)

    plog.log(
        f"Pipeline complete: {df.shape[0]:,} rows × {df.shape[1]} columns "
        f"({nulls_handled:,} nulls handled, "
        f"{duplicates_removed} duplicates removed)"
    )

    return {
        "cleaned_df": df,
        "original_shape": original_shape,
        "cleaned_shape": df.shape,
        "nulls_handled": nulls_handled,
        "duplicates_removed": duplicates_removed,
        "type_fixes": type_fixes,
        "columns_dropped": columns_dropped,
        "outliers_detected": outlier_report_kpi,
        "outlier_method": method_used,
        "skew_report": skew_report_df,
        "high_skew": high_skew,
        "skew_kpi": skew_kpi,
        "logs": list(plog),
    }
