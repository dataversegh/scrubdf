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
        result = low_var_flag(numeric_df)
        assert "constant" in result

    def test_ignores_normal_variance(self, numeric_df):
        result = low_var_flag(numeric_df)
        assert "normal" not in result

    def test_no_numeric_returns_empty(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        assert low_var_flag(df) == []


class TestMissingEda:
    def test_returns_percentages(self, missing_df):
        result = missing_eda(missing_df)
        assert "num_a" in result
        assert result["num_a"].endswith("%")

    def test_zero_for_complete_col(self, missing_df):
        result = missing_eda(missing_df)
        assert result["text_b"] == "0.0%"

    def test_empty_df_returns_zeros(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        result = missing_eda(df)
        assert result["a"] == "0.0%"


class TestGetDtypes:
    def test_returns_series(self, messy_df):
        result = get_dtypes(messy_df)
        assert isinstance(result, pd.Series)
        assert result.sum() == len(messy_df.columns)


class TestNullCols:
    def test_detects_high_null_column(self):
        df = pd.DataFrame({
            "mostly_null": [None] * 80 + [1.0] * 20,
            "fine": range(100),
        })
        result = null_cols(df)
        assert "mostly_null" in result
        assert "fine" not in result

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        assert null_cols(df) == []


class TestTopFreqCols:
    def test_returns_tidy_dataframe(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "red"]})
        result = top_freq_cols(df)
        assert set(result.columns) == {"column", "value", "frequency"}
        assert result.iloc[0]["value"] == "red"

    def test_no_object_cols_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = top_freq_cols(df)
        assert result.empty


class TestCorrMatrix:
    def test_returns_square_matrix(self, numeric_df):
        result = corr_matrix(numeric_df)
        n_numeric = len(numeric_df.select_dtypes(include="number").columns)
        assert result.shape == (n_numeric, n_numeric)

    def test_diagonal_is_one(self, numeric_df):
        result = corr_matrix(numeric_df)
        diag = np.diag(result.values)
        non_nan_diag = diag[~np.isnan(diag)]
        np.testing.assert_array_almost_equal(non_nan_diag, 1.0)

    def test_single_numeric_col_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = corr_matrix(df)
        assert result.empty


class TestCategoricalCols:
    def test_identifies_categorical(self):
        df = pd.DataFrame({
            "cat": ["a", "b", "a", "b", "a"] * 20,
            "unique": range(100),
        })
        cols, ratios = categorical_cols(df)
        assert "cat" in cols

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        cols, ratios = categorical_cols(df)
        assert cols == []
        assert ratios == {}


class TestFlagsReport:
    def test_returns_dataframe(self, messy_df):
        result = flags_report(messy_df, ["col1"], ["col2"], ["col3"], [])
        assert isinstance(result, pd.DataFrame)
        assert "Shape" in result.columns

    def test_none_when_no_flags(self, messy_df):
        result = flags_report(messy_df, [], [], [], [])
        assert result["High Null Columns"].iloc[0] == "None"
