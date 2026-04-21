"""Exploratory Data Analysis functions.

These functions *describe* a DataFrame without modifying it.
They return DataFrames, dicts, or lists - never render anything.
All functions handle edge cases (empty DataFrames, no numeric columns, etc.)
gracefully.
"""

from __future__ import annotations

import pandas as pd


def skewness_check(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Compute skewness for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple
        ``(skew_report_df, high_skew_descriptions, skew_kpi_dict)``
        where *high_skew_descriptions* lists columns with ``|skew| > 2``.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        empty_df = pd.DataFrame(columns=["Column", "Skew value"])
        return empty_df, [], {}

    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    skew_report = skewness.to_dict()
    high_skew = [
        f"{name}: {round(skew, 2)}"
        for name, skew in skew_report.items()
        if abs(skew) > 2
    ]
    skew_kpi = dict(list(skewness.head().items()))
    df_skew_report = pd.DataFrame(
        list(skew_report.items()), columns=["Column", "Skew value"]
    )
    return df_skew_report, high_skew, skew_kpi


def low_var_flag(df: pd.DataFrame, threshold: float = 1e-3) -> list[str]:
    """Return column names with standard deviation below *threshold*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Standard deviation cutoff (default ``0.001``).

    Returns
    -------
    list of str
        Column names with low variance.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return []
    stds = df[numeric_cols].std()
    return stds[stds < threshold].index.tolist()


def missing_eda(df: pd.DataFrame) -> dict[str, str]:
    """Return a dict of ``{column: 'X.X%'}`` showing missing-value percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Mapping of column name to missing percentage string.
    """
    if len(df) == 0:
        return {col: "0.0%" for col in df.columns}
    return {
        col: f"{df[col].isnull().sum() / len(df) * 100:.1f}%"
        for col in df.columns
    }


def get_dtypes(df: pd.DataFrame) -> pd.Series:
    """Return value counts of data types in the DataFrame.

    Renamed from the original ``dataype`` for clarity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Counts of each dtype present.
    """
    return df.dtypes.value_counts()


def card_cols(df: pd.DataFrame, threshold: int = 50) -> list[str]:
    """Return string columns with more than *threshold* unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : int
        Minimum unique-value count to be flagged (default ``50``).

    Returns
    -------
    list of str
        High-cardinality column names.
    """
    return [
        col
        for col in df.select_dtypes(include=["object", "string"]).columns
        if df[col].nunique() > threshold
    ]


def top_freq_cols(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return the top *n* most frequent values for each string column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n : int
        Number of top values per column (default ``3``).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns ``[column, value, frequency]``.
    """
    records: list[dict] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            for value, freq in df[col].value_counts().head(n).items():
                records.append({"column": col, "value": value, "frequency": freq})
        except Exception:
            continue
    if not records:
        return pd.DataFrame(columns=["column", "value", "frequency"])
    return pd.DataFrame(records)


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.  Returns an empty DataFrame if
        there are fewer than 2 numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def null_cols(df: pd.DataFrame, threshold: float = 0.25) -> list[str]:
    """Return columns where the fraction of nulls exceeds *threshold*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Fraction of rows that must be null to flag a column (default ``0.25``).

    Returns
    -------
    list of str
        Column names exceeding the null threshold.
    """
    if len(df) == 0:
        return []
    return [
        col for col in df.columns
        if df[col].isnull().sum() > len(df) * threshold
    ]


def categorical_cols(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, float]]:
    """Identify likely categorical columns by uniqueness ratio.

    Uses adaptive thresholds based on dataset size.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple
        ``(sorted_col_names, {col: uniqueness_ratio})``
    """
    if len(df) == 0:
        return [], {}

    threshold = 0.5 if len(df) <= 50_000 else 0.3
    max_unique = 100 if len(df) <= 50_000 else 500

    cat_col_dict: dict[str, float] = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df)
        ratio = unique_count / total_count
        if unique_count <= max_unique and ratio <= threshold:
            cat_col_dict[col] = ratio

    sorted_cols = pd.Series(cat_col_dict).sort_values().index.tolist()
    return sorted_cols, cat_col_dict


def flags_report(
    df: pd.DataFrame,
    high_null_cols: list[str],
    high_card_cols: list[str],
    low_var_cols: list[str],
    dup_cols: list[str],
) -> pd.DataFrame:
    """Build a summary flags DataFrame for display.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame (used for shape and dtype counts).
    high_null_cols : list of str
        Columns with high null percentages.
    high_card_cols : list of str
        Columns with high cardinality.
    low_var_cols : list of str
        Columns with low variance.
    dup_cols : list of str
        Duplicate column names found.

    Returns
    -------
    pd.DataFrame
        Single-row summary DataFrame.
    """
    flags_summary = {
        "High Null Columns": ", ".join(high_null_cols) if high_null_cols else "None",
        "High Cardinality Columns": ", ".join(high_card_cols) if high_card_cols else "None",
        "Low Variance Columns": ", ".join(low_var_cols) if low_var_cols else "None",
        "Duplicate Columns": ", ".join(dup_cols) if dup_cols else "None",
        "Shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "Numeric Columns": len(df.select_dtypes(include="number").columns),
        "Object Columns": len(df.select_dtypes(include=["object", "string"]).columns),
        "DateTime Columns": len(df.select_dtypes(include="datetime").columns),
    }
    return pd.DataFrame([flags_summary])