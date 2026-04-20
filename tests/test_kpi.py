"""Tests for scrubdf.kpi — formatting, edge cases, corr_pair."""

import math

import numpy as np
import pandas as pd

from scrubdf.kpi import (
    advanced_kpi_card,
    corr_pair,
    format_kpi_value,
    kpi_card,
)


class TestFormatKpiValue:
    def test_millions(self):
        assert format_kpi_value("Rows", 1_500_000) == "1.5M"

    def test_thousands(self):
        assert format_kpi_value("Rows", 2_500) == "2.5K"

    def test_percentage_label(self):
        assert format_kpi_value("Missing Values %", 12.345) == "12.3%"

    def test_small_number(self):
        assert format_kpi_value("Count", 42) == "42"

    def test_nan_returns_na(self):
        assert format_kpi_value("Test", float("nan")) == "N/A"

    def test_inf_returns_inf(self):
        assert format_kpi_value("Test", float("inf")) == "Inf"

    def test_negative_thousands(self):
        assert format_kpi_value("Delta", -3500) == "-3.5K"

    def test_zero(self):
        assert format_kpi_value("Count", 0) == "0"


class TestKpiCard:
    def test_returns_expected_keys(self):
        result = kpi_card("Total", 1234)
        assert set(result.keys()) == {"label", "raw_value", "display_value"}

    def test_raw_value_preserved(self):
        result = kpi_card("Total", 1234)
        assert result["raw_value"] == 1234

    def test_display_value_formatted(self):
        result = kpi_card("Total", 1234)
        assert result["display_value"] == "1.2K"


class TestCorrPair:
    def test_finds_high_correlation(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),  # perfectly correlated
            "c": np.random.randn(100),
        })
        corr_dict, count = corr_pair(df, threshold=0.9)
        assert count >= 1
        assert "a vs b" in corr_dict
        assert corr_dict["a vs b"] == 1.0

    def test_no_correlations_below_threshold(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        _, count = corr_pair(df, threshold=0.9)
        assert count == 0

    def test_handles_constant_column(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [5, 5, 5],  # constant — correlation is NaN
        })
        corr_dict, count = corr_pair(df)
        assert count == 0  # NaN correlation should not be flagged


class TestAdvancedKpiCard:
    def test_returns_expected_keys(self):
        result = advanced_kpi_card("Outlier Columns", 3, {"age": 12})
        expected = {"label", "raw_value", "display_value", "description", "details"}
        assert set(result.keys()) == expected

    def test_outlier_details_format(self):
        result = advanced_kpi_card("Outlier Columns", 2, {"age": 12, "income": 5})
        assert result["details"][0]["value"] == "12 outliers"

    def test_no_details_returns_empty_list(self):
        result = advanced_kpi_card("Test", 0, None)
        assert result["details"] == []

    def test_description_for_known_label(self):
        result = advanced_kpi_card("Highly Skewed Columns", 4)
        assert "Skewness" in result["description"]
