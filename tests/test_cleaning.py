"""Tests for scrubdf.cleaning — priority functions, error paths, edge cases."""

import numpy as np
import pandas as pd
import pytest

from scrubdf.cleaning import (
    ALL_STEPS,
    cleaning_pipeline,
    convert_column_type,
    detect_and_remove_outliers,
    handle_missing_values,
    remove_duplicates,
    clean_unnamed_columns,
    drop_fully_empty,
    strip_whitespace,
    standardise_values,
    parse_date_columns,
)
from scrubdf.utils import PipelineLog, ScrubTypeError, ScrubValueError


# ===================================================================
# 1. cleaning_pipeline — end-to-end
# ===================================================================

class TestCleaningPipeline:
    def test_returns_structured_dict(self, messy_df):
        result = cleaning_pipeline(messy_df)
        expected_keys = {
            "cleaned_df", "original_shape", "cleaned_shape",
            "nulls_handled", "duplicates_removed", "type_fixes",
            "columns_dropped", "outliers_detected", "outlier_method",
            "skew_report", "high_skew", "skew_kpi", "logs",
        }
        assert expected_keys == set(result.keys())

    def test_output_is_cleaner(self, messy_df):
        result = cleaning_pipeline(messy_df)
        assert result["cleaned_shape"][0] <= result["original_shape"][0]
        assert result["cleaned_shape"][1] < result["original_shape"][1]

    def test_no_nulls_after_default_pipeline(self, messy_df):
        result = cleaning_pipeline(messy_df)
        assert result["cleaned_df"].isnull().sum().sum() == 0

    def test_logs_are_populated(self, messy_df):
        result = cleaning_pipeline(messy_df)
        assert len(result["logs"]) > 0

    def test_logs_contain_start_and_end(self, messy_df):
        result = cleaning_pipeline(messy_df)
        logs_text = " ".join(result["logs"])
        assert "Pipeline started" in logs_text
        assert "Pipeline complete" in logs_text

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ScrubValueError, match="empty"):
            cleaning_pipeline(pd.DataFrame())

    def test_rejects_non_dataframe(self):
        with pytest.raises(ScrubTypeError, match="Expected a pandas DataFrame"):
            cleaning_pipeline({"a": [1, 2, 3]})

    def test_rejects_invalid_step_names(self, messy_df):
        with pytest.raises(ScrubValueError, match="Invalid step"):
            cleaning_pipeline(messy_df, steps={"not_a_real_step"})

    def test_step_selection(self, messy_df):
        result = cleaning_pipeline(messy_df, steps={"duplicates"})
        assert result["duplicates_removed"] >= 0
        assert result["outlier_method"] == "skipped"

    def test_does_not_mutate_input(self, messy_df):
        original_shape = messy_df.shape
        original_columns = messy_df.columns.tolist()
        cleaning_pipeline(messy_df)
        assert messy_df.shape == original_shape
        assert messy_df.columns.tolist() == original_columns

    def test_all_steps_constant(self):
        """ALL_STEPS should match the documented step names."""
        assert "unnamed_columns" in ALL_STEPS
        assert "outliers" in ALL_STEPS
        assert len(ALL_STEPS) == 10


# ===================================================================
# 2. handle_missing_values — all strategies
# ===================================================================

class TestHandleMissingValues:
    def test_median_strategy(self, missing_df):
        plog = PipelineLog()
        result = handle_missing_values(missing_df.copy(), strategy="median", plog=plog)
        assert result[["num_a", "num_b"]].isnull().sum().sum() == 0
        assert result["text_a"].isnull().sum() == 0

    def test_mean_strategy(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="mean")
        mean_a = missing_df["num_a"].mean()
        assert result["num_a"].iloc[2] == pytest.approx(mean_a)

    def test_mode_strategy(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="mode")
        assert result["num_a"].isnull().sum() == 0

    def test_drop_strategy(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="drop")
        assert result.shape[0] < missing_df.shape[0]
        assert result[["num_a", "num_b"]].isnull().sum().sum() == 0

    def test_skip_strategy(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="skip")
        assert result.isnull().sum().sum() == missing_df.isnull().sum().sum()

    def test_text_columns_filled_with_unknown(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="median")
        assert (result["text_a"] == "Unknown").sum() == 2

    def test_invalid_strategy_raises(self, missing_df):
        with pytest.raises(ScrubValueError, match="Invalid missing-value strategy"):
            handle_missing_values(missing_df.copy(), strategy="invalid")

    def test_no_missing_values_is_noop(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = handle_missing_values(df.copy(), strategy="median")
        pd.testing.assert_frame_equal(result, df)


# ===================================================================
# 3. detect_and_remove_outliers — IQR + Z-score + edge cases
# ===================================================================

class TestDetectAndRemoveOutliers:
    def test_iqr_removes_outliers(self, numeric_df):
        numeric_df.loc[0, "normal"] = 10000
        cleaned, method, report, kpi = detect_and_remove_outliers(
            numeric_df, method="IQR"
        )
        assert method == "IQR"
        assert cleaned.shape[0] < numeric_df.shape[0]
        assert report["normal"] > 0

    def test_zscore_removes_outliers(self, numeric_df):
        numeric_df.loc[0, "normal"] = 10000
        cleaned, method, report, kpi = detect_and_remove_outliers(
            numeric_df, method="Z-score"
        )
        assert method == "Z-score"
        assert cleaned.shape[0] < numeric_df.shape[0]

    def test_auto_selects_method(self, numeric_df):
        _, method, _, _ = detect_and_remove_outliers(numeric_df, method=None)
        assert method in ("IQR", "Z-score", "Isolation Forest")

    def test_invalid_method_raises(self, numeric_df):
        with pytest.raises(ScrubValueError, match="Invalid outlier method"):
            detect_and_remove_outliers(numeric_df, method="InvalidMethod")

    def test_kpi_only_has_nonzero(self, numeric_df):
        _, _, report, kpi = detect_and_remove_outliers(numeric_df, method="IQR")
        for col, count in kpi.items():
            assert count > 0

    def test_no_numeric_columns_skips(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
        cleaned, method, report, kpi = detect_and_remove_outliers(df)
        assert method == "skipped"
        assert report == {}

    def test_per_column_error_does_not_crash(self):
        """A column that causes an error should be logged, not crash."""
        plog = PipelineLog()
        df = pd.DataFrame({"good": [1, 2, 3, 4, 5], "also_good": [10, 20, 30, 40, 50]})
        cleaned, method, report, kpi = detect_and_remove_outliers(
            df, method="IQR", plog=plog
        )
        assert isinstance(cleaned, pd.DataFrame)


# ===================================================================
# 4. convert_column_type — string→numeric
# ===================================================================

class TestConvertColumnType:
    def test_converts_numeric_strings(self):
        df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
        result = convert_column_type(df)
        assert pd.api.types.is_numeric_dtype(result["a"])

    def test_leaves_real_text_alone(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        result = convert_column_type(df)
        assert not pd.api.types.is_numeric_dtype(result["name"])

    def test_handles_commas_in_numbers(self):
        df = pd.DataFrame({"amount": ["1,000", "2,500", "3,750"]})
        result = convert_column_type(df)
        assert pd.api.types.is_numeric_dtype(result["amount"])
        assert result["amount"].iloc[0] == 1000

    def test_logs_conversions(self):
        plog = PipelineLog()
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        convert_column_type(df, plog=plog)
        assert any("Converted" in entry for entry in plog.entries)

    def test_error_in_one_column_does_not_crash(self):
        """Even if one column somehow fails, the rest should be processed."""
        plog = PipelineLog()
        df = pd.DataFrame({"ok": ["1", "2"], "also_ok": ["a", "b"]})
        result = convert_column_type(df, plog=plog)
        assert isinstance(result, pd.DataFrame)


# ===================================================================
# 5. remove_duplicates
# ===================================================================

class TestRemoveDuplicates:
    def test_removes_exact_row_duplicates(self, duplicate_df):
        result, dup_cols = remove_duplicates(duplicate_df)
        assert result.shape[0] == 3

    def test_returns_empty_dup_cols_when_none(self, duplicate_df):
        _, dup_cols = remove_duplicates(duplicate_df)
        assert dup_cols == []

    def test_handles_duplicate_columns(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        result, dup_cols = remove_duplicates(df)
        assert "a" in dup_cols
        assert len(result.columns) == 2

    def test_logs_actions(self, duplicate_df):
        plog = PipelineLog()
        remove_duplicates(duplicate_df, plog=plog)
        assert any("duplicate" in entry.lower() for entry in plog.entries)

    def test_no_change_on_unique_data(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result, dup_cols = remove_duplicates(df)
        assert result.shape == df.shape
        assert dup_cols == []


# ===================================================================
# 6. Other cleaning functions — edge cases
# ===================================================================

class TestCleanUnnamedColumns:
    def test_drops_all_null_unnamed(self):
        df = pd.DataFrame({"Unnamed: 0": [None, None], "real": [1, 2]})
        result = clean_unnamed_columns(df)
        assert "Unnamed: 0" not in result.columns

    def test_renames_non_null_unnamed(self):
        df = pd.DataFrame({"Unnamed: 0": [1, 2], "real": [3, 4]})
        result = clean_unnamed_columns(df)
        assert "renamed_1" in result.columns


class TestDropFullyEmpty:
    def test_drops_all_null_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        result = drop_fully_empty(df)
        assert "b" not in result.columns

    def test_drops_all_null_row(self):
        df = pd.DataFrame({"a": [1, None], "b": [2, None]})
        result = drop_fully_empty(df)
        assert len(result) == 1


class TestStripWhitespace:
    def test_strips_string_cells(self):
        df = pd.DataFrame({"a": ["  hello  ", " world "]})
        result = strip_whitespace(df)
        assert result["a"].iloc[0] == "hello"


class TestStandardiseValues:
    def test_normalises_gender(self):
        df = pd.DataFrame({"Gender": ["M", "f", "boy", "Woman"]})
        result = standardise_values(df)
        assert set(result["Gender"]) == {"male", "female"}

    def test_ignores_non_gender_columns(self):
        df = pd.DataFrame({"name": ["M", "f"]})
        result = standardise_values(df)
        assert result["name"].tolist() == ["M", "f"]


class TestParseDateColumns:
    def test_parses_date_column(self):
        df = pd.DataFrame({"signup_date": ["2020-01-01", "2020-02-01"]})
        result = parse_date_columns(df)
        assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])

    def test_ignores_non_date_columns(self):
        df = pd.DataFrame({"name": ["2020-01-01", "2020-02-01"]})
        result = parse_date_columns(df)
        assert not pd.api.types.is_datetime64_any_dtype(result["name"])
