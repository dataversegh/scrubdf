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
    detect_id_columns,
    is_string_dtype,
    modified_z,
    normalize_unicode,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def clean_unnamed_columns(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Drop all-null unnamed/blank columns; rename non-null ones."""
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
    """Standardise column names: lowercase, strip, replace spaces with underscores."""
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
    """Strip leading/trailing whitespace from all string cells."""
    try:
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        if plog:
            plog.log("Stripped whitespaces from string entries")
    except Exception as e:
        if plog:
            plog.error(f"Error stripping whitespace: {e}")
    return df


def normalize_encoding_step(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Normalize unicode in string columns (curly quotes, em dashes, etc.)."""
    count = 0
    for col in df.columns:
        try:
            if is_string_dtype(df[col].dtype):
                original = df[col].copy()
                df[col] = df[col].map(
                    lambda x: normalize_unicode(x) if isinstance(x, str) else x
                )
                if not df[col].equals(original):
                    count += 1
        except Exception as e:
            if plog:
                plog.error(f"Error normalizing encoding in '{col}': {e}")
    if plog and count > 0:
        plog.log(f"Normalized unicode encoding in {count} columns")
    return df


def convert_column_type(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
    exclude_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Attempt to convert string columns to numeric where possible.

    Parameters
    ----------
    exclude_cols : set of str, optional
        Columns to skip (e.g. detected ID columns).
    """
    skip = exclude_cols or set()
    for col in df.columns:
        if col in skip:
            continue
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

    Domain-specific to survey/demographic data.
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
    """Remove duplicate columns and duplicate rows."""
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        df = df.loc[:, ~df.columns.duplicated()]
        if plog:
            plog.log(f"Dropped duplicate columns: {dup_cols}")

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
    """Drop columns and rows that are entirely null."""
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


def drop_high_null_columns(
    df: pd.DataFrame,
    threshold: float = 0.7,
    protect_cols: set[str] | None = None,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Drop columns where the fraction of nulls exceeds *threshold*.

    Parameters
    ----------
    threshold : float
        Drop columns with null fraction above this (default 0.7 = 70%).
    protect_cols : set of str, optional
        Columns to never drop (e.g. skip-logic columns in survey data).
    """
    protected = protect_cols or set()
    dropped: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return df

    for col in df.columns:
        if col in protected:
            continue
        null_frac = df[col].isnull().sum() / n_rows
        if null_frac > threshold:
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)
        if plog:
            plog.log(
                f"Dropped {len(dropped)} high-null columns "
                f"(>{threshold:.0%} null): {dropped}"
            )

    return df


def drop_constant_columns(
    df: pd.DataFrame,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Drop columns where every value is identical (zero variance)."""
    constant_cols: list[str] = []
    for col in df.columns:
        try:
            if df[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)
        except Exception:
            continue

    if constant_cols:
        df = df.drop(columns=constant_cols)
        if plog:
            plog.log(f"Dropped constant columns (zero variance): {constant_cols}")

    return df


_VALID_MISSING_STRATEGIES = {"median", "mean", "mode", "drop", "skip"}


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal["median", "mean", "mode", "drop", "skip"] = "median",
    exclude_cols: set[str] | None = None,
    plog: PipelineLog | None = None,
) -> pd.DataFrame:
    """Handle missing values in numeric and text columns.

    Parameters
    ----------
    strategy : str
        How to handle numeric nulls: 'median', 'mean', 'mode', 'drop', 'skip'.
    exclude_cols : set of str, optional
        Columns to leave untouched (e.g. skip-logic columns).
    """
    if strategy not in _VALID_MISSING_STRATEGIES:
        raise ScrubValueError(
            f"Invalid missing-value strategy '{strategy}'. "
            f"Choose from: {sorted(_VALID_MISSING_STRATEGIES)}"
        )

    skip = exclude_cols or set()

    if strategy == "skip":
        if plog:
            plog.log("Skipped missing value treatment")
        return df

    if strategy == "drop":
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns if c not in skip
        ]
        before = df.shape[0]
        df = df.dropna(subset=numeric_cols)
        if plog:
            plog.log(f"Dropped {before - df.shape[0]} rows with missing numeric values")
    else:
        for col in df.select_dtypes(include="number").columns:
            if col in skip:
                continue
            try:
                num_missing = df[col].isnull().sum()
                if num_missing > 0:
                    if strategy == "median":
                        if pd.api.types.is_integer_dtype(df[col]):
                            fill_val = int(round(df[col].median()))
                        fill_val = df[col].median()
                    elif strategy == "mean":
                        if pd.api.types.is_integer_dtype(df[col]):
                            fill_val = int(round(df[col].mean()))
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
        if col in skip:
            continue
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
    """Auto-detect and parse date/timestamp columns."""
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
    mode: Literal["remove", "flag", "skip"] = "remove",
    exclude_cols: set[str] | None = None,
    plog: PipelineLog | None = None,
) -> tuple[pd.DataFrame, str, dict[str, int], dict[str, int]]:
    """Detect outliers with option to remove, flag, or skip.

    Parameters
    ----------
    method : str or None
        Detection method. None = auto-select.
    mode : str
        ``'remove'`` drops outlier rows (default).
        ``'flag'`` adds ``_is_outlier`` boolean column instead of dropping.
        ``'skip'`` does nothing.
    exclude_cols : set of str, optional
        Numeric columns to exclude from outlier detection
        (e.g. ID columns, ordinal columns).
    """
    if mode == "skip":
        if plog:
            plog.log("Outlier detection skipped")
        return df, "skipped", {}, {}

    skip = exclude_cols or set()
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns.tolist()
        if c not in skip
    ]
    outlier_report: dict[str, int] = {}

    if not numeric_cols:
        if plog:
            plog.warn("No eligible numeric columns — skipping outlier detection")
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

    if mode == "flag":
        # Flag mode: add boolean column, don't remove rows
        outlier_mask = pd.Series(False, index=df.index)

        if method == "IQR":
            for col in numeric_cols:
                try:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    col_outliers = (df[col] > q3 + 1.5 * iqr) | (df[col] < q1 - 1.5 * iqr)
                    outlier_report[col] = int(col_outliers.sum())
                    outlier_mask |= col_outliers
                except Exception as e:
                    if plog:
                        plog.error(f"IQR failed on column '{col}': {e}")

        elif method == "Z-score":
            for col in numeric_cols:
                try:
                    mz = modified_z(df[col])
                    col_outliers = np.abs(mz) > 3.5
                    outlier_report[col] = int(col_outliers.sum())
                    outlier_mask |= col_outliers
                except Exception as e:
                    if plog:
                        plog.error(f"Z-score failed on column '{col}': {e}")

        elif method == "Isolation Forest":
            try:
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination="auto", random_state=42)
                preds = iso.fit_predict(df[numeric_cols])
                outlier_mask = pd.Series(preds == -1, index=df.index)
                outlier_report["Total"] = int(outlier_mask.sum())
            except ImportError:
                raise ImportError(
                    "Isolation Forest requires scikit-learn. "
                    "Install with: pip install scrubdf[ml]"
                )
            except Exception as e:
                if plog:
                    plog.error(f"Isolation Forest failed: {e}")

        df["_is_outlier"] = outlier_mask
        if plog:
            plog.log(
                f"Outlier detection using {method} (flag mode): "
                f"{outlier_mask.sum()} rows flagged"
            )

    else:
        # Remove mode (original behavior)
        cleaned_df = df.copy()

        if method == "IQR":
            for col in numeric_cols:
                try:
                    q1 = cleaned_df[col].quantile(0.25)
                    q3 = cleaned_df[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = (cleaned_df[col] > q3 + 1.5 * iqr) | (cleaned_df[col] < q1 - 1.5 * iqr)
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
                iso = IsolationForest(contamination="auto", random_state=42)
                preds = iso.fit_predict(cleaned_df[numeric_cols])
                mask = preds == 1
                outlier_report["Total"] = int((~mask).sum())
                cleaned_df = cleaned_df[mask]
            except ImportError:
                raise ImportError(
                    "Isolation Forest requires scikit-learn. "
                    "Install with: pip install scrubdf[ml]"
                )
            except Exception as e:
                if plog:
                    plog.error(f"Isolation Forest failed: {e}")

        if plog:
            removed = df.shape[0] - cleaned_df.shape[0]
            plog.log(f"Outlier detection using {method}: {removed} rows removed")
            affected = [k for k, v in outlier_report.items() if isinstance(v, int) and v > 0]
            if affected:
                plog.log(f"Outliers detected in columns: {affected}")

        df = cleaned_df

    outlier_report_kpi = {
        k: v for k, v in outlier_report.items()
        if isinstance(v, int) and v != 0
    }

    return df, method, outlier_report, outlier_report_kpi


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
    "drop_high_null",
    "drop_constant",
    "normalize_encoding",
})


def cleaning_pipeline(
    df: pd.DataFrame,
    *,
    profile: Literal["auto", "transactional", "survey"] | None = None,
    steps: Set[str] | None = None,
    missing_strategy: Literal["median", "mean", "mode", "drop", "skip"] | None = None,
    outlier_method: Literal["IQR", "Z-score", "Isolation Forest"] | None = None,
    outlier_mode: Literal["remove", "flag", "skip"] | None = None,
) -> Dict[str, Any]:
    """Run the full cleaning pipeline and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input DataFrame.  A copy is made internally.
    profile : str, optional
        Data profile preset: ``'auto'``, ``'transactional'``, or ``'survey'``.
        Profiles set sensible defaults for steps, strategies, and thresholds.
        Explicit parameters (steps, missing_strategy, etc.) override
        profile defaults.
    steps : set of str, optional
        Which cleaning steps to run.  Overrides profile.
    missing_strategy : str, optional
        Strategy for handling missing values.  Overrides profile.
    outlier_method : str or None, optional
        Outlier detection method.  Overrides profile.
    outlier_mode : str, optional
        ``'remove'``, ``'flag'``, or ``'skip'``.  Overrides profile.

    Returns
    -------
    dict
        Structured report with keys: cleaned_df, original_shape,
        cleaned_shape, nulls_handled, duplicates_removed, type_fixes,
        columns_dropped, outliers_detected, outlier_method, skew_report,
        high_skew, skew_kpi, id_columns, protected_columns, profile_used, logs.
    """
    validate_dataframe(df)
    df = df.copy()

    # --- Resolve profile ---
    from scrubdf.profiles import (
        get_profile,
        detect_ordinal_columns,
        detect_coded_categoricals,
        detect_skip_logic_columns,
        PROFILE_AUTO,
    )

    if profile is not None:
        prof = get_profile(profile)
    else:
        prof = PROFILE_AUTO

    active_steps = steps if steps is not None else prof.steps
    miss_strat = missing_strategy if missing_strategy is not None else prof.missing_strategy
    out_mode = outlier_mode if outlier_mode is not None else prof.outlier_mode

    # Validate step names
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
        f"Pipeline started ({prof.name} profile): {original_shape[0]:,} rows x "
        f"{original_shape[1]} columns, {original_nulls:,} null values"
    )

    # --- Detect special columns ---
    id_cols: set[str] = set()
    if prof.protect_id_columns:
        id_cols = set(detect_id_columns(df))
        if id_cols and plog:
            plog.log(f"Detected ID columns (excluded from numeric analysis): {sorted(id_cols)}")

    ordinal_cols: set[str] = set()
    if prof.protect_ordinal_columns:
        ordinal_cols = set(detect_ordinal_columns(df, max_unique=prof.coded_categorical_max))
        if ordinal_cols and plog:
            plog.log(f"Detected ordinal/Likert columns: {sorted(ordinal_cols)}")

    skip_logic_cols: set[str] = set()
    if prof.respect_skip_logic:
        skip_logic_cols = set(detect_skip_logic_columns(df))
        if skip_logic_cols and plog:
            plog.log(f"Detected skip-logic columns (missingness preserved): {sorted(skip_logic_cols)}")

    coded_cats: set[str] = set()
    if prof.detect_coded_categoricals:
        coded_cats = set(detect_coded_categoricals(df, max_unique=prof.coded_categorical_max))
        coded_cats -= id_cols  # don't double-count
        if coded_cats and plog:
            plog.log(f"Detected coded categorical columns: {sorted(coded_cats)}")

    # Columns protected from numeric operations
    protected_from_numeric = id_cols | ordinal_cols | coded_cats
    # Columns protected from missing value imputation
    protected_from_imputation = skip_logic_cols

    # --- Prep & formatting ---
    if "unnamed_columns" in active_steps:
        df = clean_unnamed_columns(df, plog)
    if "column_names" in active_steps:
        df = clean_col_name(df, plog)
        # Update protected sets after column name changes
        id_cols = {c.strip().lower().replace(" ", "_") for c in id_cols}
        ordinal_cols = {c.strip().lower().replace(" ", "_") for c in ordinal_cols}
        skip_logic_cols = {c.strip().lower().replace(" ", "_") for c in skip_logic_cols}
        coded_cats = {c.strip().lower().replace(" ", "_") for c in coded_cats}
        protected_from_numeric = id_cols | ordinal_cols | coded_cats
        protected_from_imputation = skip_logic_cols
    if "whitespace" in active_steps:
        df = strip_whitespace(df, plog)
    if "normalize_encoding" in active_steps:
        df = normalize_encoding_step(df, plog)
    if "type_conversion" in active_steps:
        before_dtypes = df.dtypes.to_dict()
        df = convert_column_type(df, plog, exclude_cols=protected_from_numeric)
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
    if "drop_high_null" in active_steps and prof.drop_high_null_threshold is not None:
        df = drop_high_null_columns(
            df,
            threshold=prof.drop_high_null_threshold,
            protect_cols=protected_from_imputation,
            plog=plog,
        )
    if "drop_constant" in active_steps and prof.drop_constant_columns:
        df = drop_constant_columns(df, plog)
    if "duplicates" in active_steps:
        before_dup = df.shape[0]
        df, dup_cols = remove_duplicates(df, plog)
        duplicates_removed = before_dup - df.shape[0]
    else:
        duplicates_removed = 0
    if "missing_values" in active_steps:
        df = handle_missing_values(
            df,
            strategy=miss_strat,
            exclude_cols=protected_from_imputation,
            plog=plog,
        )
    if "date_parsing" in active_steps:
        df = parse_date_columns(df, plog)

    # --- Quality checks ---
    plog.log(f"Before outlier step: rows={len(df)}, cols={df.columns.tolist()}")
    if "outliers" in active_steps:
        df, method_used, outlier_report, outlier_report_kpi = (
            detect_and_remove_outliers(
                df,
                method=outlier_method,
                mode=out_mode,
                exclude_cols=protected_from_numeric,
                plog=plog,
            )
        )
    else:
        method_used = "skipped"
        outlier_report = {}
        outlier_report_kpi = {}
    
    plog.log(f"After outlier step: rows={len(df)}, cols={df.columns.tolist()}")
    plog.log(f"_is_outlier present? {'_is_outlier' in df.columns}")

    # --- Post-clean analysis ---
    remaining_nulls = int(df.isnull().sum().sum())
    nulls_handled = original_nulls - remaining_nulls
    columns_dropped = sorted(original_cols - set(df.columns))

    from scrubdf.eda import skewness_check
    # Exclude ID/ordinal columns from skewness analysis
    analysis_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in protected_from_numeric and c != "_is_outlier"
    ]
    if analysis_cols:
        skew_report_df, high_skew, skew_kpi = skewness_check(df[analysis_cols])
    else:
        skew_report_df = pd.DataFrame(columns=["Column", "Skew value"])
        high_skew = []
        skew_kpi = {}

    plog.log(
        f"Pipeline complete: {df.shape[0]:,} rows x {df.shape[1]} columns "
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
        "id_columns": sorted(id_cols & set(df.columns)),
        "protected_columns": sorted(
            (protected_from_numeric | protected_from_imputation) & set(df.columns)
        ),
        "profile_used": prof.name,
        "logs": list(plog),
    }