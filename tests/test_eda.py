"""Tests for scrubdf.eda — covers all EDA functions including edge cases."""

import numpy as np
import pandas as pd
import pytest

from scrubdf.eda import (
    categorical_cols,
    corr_matrix,
    flags_report,
    get_dtypes,
    low_var_flag,
    missing_eda,
    null_cols,
    skewness_check,
    top_freq_cols,
)


class TestSkewnessCheck:
    def test_returns_three_items(self, numeric_df):
        report, high_skew, kpi = skewness_check(numeric_df)
        assert isinstance(report, pd.DataFrame)
        assert isinstance(high_skew, list)
        assert isinstance(kpi, dict)

    def test_detects_skewed_column(self, numeric_df):
        _, high_skew, _ = skewness_check(numeric_df)
        skew_text = " ".join(high_skew)
        assert "skewed" in skew_text or len(high_skew) > 0

    def test_no_numeric_cols_returns_empty(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["m", "n"]})
        report, high_skew, kpi = skewness_check(df)
        assert report.empty
        assert high_skew == []
        assert kpi == {}


class TestLowVarFlag:
    def test_detects_constant_column(self, numeric_df):
        assert "constant" in low_var_flag(numeric_df)

    def test_ignores_normal_variance(self, numeric_df):
        assert "normal" not in low_var_flag(numeric_df)

    def test_no_numeric_returns_empty(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        assert low_var_flag(df) == []


class TestMissingEda:
    def test_returns_percentages(self, missing_df):
        result = missing_eda(missing_df)
        assert result["num_a"].endswith("%")

    def test_zero_for_complete_col(self, missing_df):
        assert missing_eda(missing_df)["text_b"] == "0.0%"


class TestGetDtypes:
    def test_returns_series(self, messy_df):
        result = get_dtypes(messy_df)
        assert isinstance(result, pd.Series)


class TestNullCols:
    def test_detects_high_null_column(self):
        df = pd.DataFrame({
            "mostly_null": [None] * 80 + [1.0] * 20,
            "fine": range(100),
        })
        result = null_cols(df)
        assert "mostly_null" in result
        assert "fine" not in result


class TestTopFreqCols:
    def test_returns_tidy_dataframe(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "red"]})
        result = top_freq_cols(df)
        assert set(result.columns) == {"column", "value", "frequency"}

    def test_no_object_cols_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert top_freq_cols(df).empty


class TestCorrMatrix:
    def test_returns_square_matrix(self, numeric_df):
        result = corr_matrix(numeric_df)
        n = len(numeric_df.select_dtypes(include="number").columns)
        assert result.shape == (n, n)

    def test_diagonal_is_one(self, numeric_df):
        result = corr_matrix(numeric_df)
        diag = np.diag(result.values)
        non_nan = diag[~np.isnan(diag)]
        np.testing.assert_array_almost_equal(non_nan, 1.0)

    def test_single_col_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert corr_matrix(df).empty


class TestCategoricalCols:
    def test_identifies_categorical(self):
        df = pd.DataFrame({
            "cat": ["a", "b", "a", "b", "a"] * 20,
            "unique": range(100),
        })
        cols, _ = categorical_cols(df)
        assert "cat" in cols


class TestFlagsReport:
    def test_returns_dataframe(self, messy_df):
        result = flags_report(messy_df, ["c1"], ["c2"], ["c3"], [])
        assert isinstance(result, pd.DataFrame)
        assert "Shape" in result.columns