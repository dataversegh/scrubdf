"""Shared test fixtures for scrubdf."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_df():
    """A deliberately messy DataFrame that exercises every cleaning step."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "Unnamed: 0": range(n),
        "Name": [f"Person {i}" for i in range(n)],
        " Age ": np.random.randint(18, 80, n).astype(float),
        "Income": np.random.normal(50000, 15000, n),
        "Gender": np.random.choice(
            ["M", "f", "Male", "female", "boy", "Woman", "n/a"], n
        ),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "Score": ["85", "90", "78", "92", "88"] * 20,
        "empty_col": [None] * n,
    })

    df.loc[0:4, " Age "] = np.nan
    df.loc[10:14, "Income"] = np.nan
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    df.loc[0, "Income"] = 999999

    return df


@pytest.fixture
def numeric_df():
    """A simple numeric DataFrame for outlier/skewness tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "normal": np.random.normal(100, 10, 200),
        "skewed": np.random.exponential(1, 200) ** 2,
        "constant": [42.0] * 200,
    })


@pytest.fixture
def missing_df():
    """DataFrame with various missing value patterns."""
    return pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0, np.nan],
        "num_b": [10.0, np.nan, 30.0, np.nan, 50.0],
        "text_a": ["hello", None, "world", None, "test"],
        "text_b": ["a", "b", "c", "d", "e"],
    })


@pytest.fixture
def duplicate_df():
    """DataFrame with exact duplicate rows."""
    return pd.DataFrame({
        "a": [1, 2, 3, 1, 2],
        "b": [10, 20, 30, 10, 20],
        "c": ["x", "y", "z", "x", "y"],
    })


@pytest.fixture
def id_df():
    """DataFrame with ID columns that should be detected and protected."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "student_id": range(1001, 1001 + n),
        "respondent_code": [f"R{i:04d}" for i in range(n)],
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n),
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], n),
    })


@pytest.fixture
def survey_df():
    """DataFrame simulating survey data with Likert scales and skip logic."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "age": np.random.randint(18, 70, n),
        "gender": np.random.choice([1, 2, 3], n),  # coded: 1=M, 2=F, 3=Other
        "q1_satisfaction": np.random.choice([1, 2, 3, 4, 5], n),
        "q2_recommend": np.random.choice([1, 2, 3, 4, 5], n),
        "q3_used_product": np.random.choice([0, 1], n),  # 0=No, 1=Yes
        "q4_product_rating": np.random.choice([1, 2, 3, 4, 5], n),
        "q5_comments": np.random.choice(
            ["Great", "OK", "Bad", "Amazing", None], n
        ),
        "income_bracket": np.random.choice([1, 2, 3, 4, 5], n),
    })

    # Simulate skip logic: q4 only answered if q3=1
    df.loc[df["q3_used_product"] == 0, "q4_product_rating"] = np.nan

    return df


@pytest.fixture
def tmp_csv(tmp_path):
    """Write a small CSV file and return its path."""
    p = tmp_path / "test.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_csv(p, index=False)
    return p