"""KPI computation functions.

These return plain Python dicts — the rendering layer (Streamlit, React,
email template) decides how to display them.  All functions validate
inputs and handle edge cases (NaN, negative values, zero).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Optional

import pandas as pd


def format_kpi_value(label: str, value: float) -> str:
    """Format a numeric KPI value for display (e.g. ``1500`` → ``'1.5K'``).

    Parameters
    ----------
    label : str
        KPI label — used to determine special formatting rules
        (e.g. percentage suffix for "Missing Values %").
    value : float
        Raw numeric value.

    Returns
    -------
    str
        Human-readable formatted value.
    """
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "N/A"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"

    if abs(value) >= 1_000_000:
        formatted = value / 1_000_000
        return f"{round(formatted, 1)}M"
    elif abs(value) >= 1_000:
        formatted = value / 1_000
        return f"{round(formatted, 1)}K"
    elif label.lower() == "missing values %":
        return f"{round(value, 1)}%"
    else:
        return str(value)


def kpi_card(label: str, value: float) -> Dict[str, Any]:
    """Compute a single KPI card's data.

    Parameters
    ----------
    label : str
        Display label for the KPI.
    value : float
        Raw numeric value.

    Returns
    -------
    dict
        ``{"label": str, "raw_value": float, "display_value": str}``
    """
    return {
        "label": label,
        "raw_value": value,
        "display_value": format_kpi_value(label, value),
    }


def corr_pair(
    df: pd.DataFrame, threshold: float = 0.9,
) -> tuple[Dict[str, float], int]:
    """Find highly correlated numeric column pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Absolute correlation threshold (default ``0.9``).

    Returns
    -------
    tuple
        ``(corr_dict, count)`` where *corr_dict* maps
        ``'Col1 vs Col2'`` → correlation value, and *count* is the
        number of pairs found.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr_dict: Dict[str, float] = {}

    for col1, col2 in itertools.combinations(numeric_cols, 2):
        try:
            corr = df[col1].corr(df[col2])
            if pd.notna(corr) and abs(corr) > threshold:
                corr_dict[f"{col1} vs {col2}"] = round(corr, 3)
        except Exception:
            continue

    return corr_dict, len(corr_dict)


# Descriptions for KPI categories — useful for tooltips in frontends
KPI_DESCRIPTIONS: Dict[str, str] = {
    "Highly Skewed Columns": (
        "Skewness shows whether data tails more to the left (-) "
        "or right (+) of the mean"
    ),
    "Highly Correlated Columns": (
        "Correlation measures the strength and direction of a linear "
        "relationship between two variables "
        "(-1 = negative, +1 = positive, 0 = none)"
    ),
    "Outlier Columns": (
        "Outliers are data points that deviate significantly from the "
        "overall pattern, indicating unusual or extreme values"
    ),
}


def advanced_kpi_card(
    label: str,
    value: float,
    details: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute an advanced KPI card with detail breakdown.

    Parameters
    ----------
    label : str
        Display label.
    value : float
        Raw numeric value.
    details : dict, optional
        Breakdown data (e.g. ``{"age": 12, "income": 19}`` for outliers).

    Returns
    -------
    dict
        ``{"label", "raw_value", "display_value", "description", "details"}``
    """
    description = KPI_DESCRIPTIONS.get(label, "")
    detail_list: list[Dict[str, str]] = []

    if details:
        for key, val in details.items():
            try:
                if label == "Outlier Columns":
                    detail_list.append(
                        {"name": str(key), "value": f"{val} outliers"}
                    )
                else:
                    detail_list.append(
                        {"name": str(key), "value": str(round(float(val), 1))}
                    )
            except (ValueError, TypeError):
                detail_list.append({"name": str(key), "value": str(val)})

    return {
        "label": label,
        "raw_value": value,
        "display_value": format_kpi_value(label, value),
        "description": description,
        "details": detail_list,
    }
