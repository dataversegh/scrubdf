"""Tests for scrubdf.cleaning - all functions, profiles, edge cases."""

import numpy as np
import pandas as pd
import pytest

from scrubdf.cleaning import (
    ALL_STEPS,
    cleaning_pipeline,
    convert_column_type,
    detect_and_remove_outliers,
    drop_constant_columns,
    drop_high_null_columns,
    handle_missing_values,
    normalize_encoding_step,
    remove_duplicates,
    clean_unnamed_columns,
    drop_fully_empty,
    strip_whitespace,
    standardise_values,
    parse_date_columns,
)
from scrubdf.utils import PipelineLog, ScrubTypeError, ScrubValueError


# ===================================================================
# 1. cleaning_pipeline — end-to-end + profiles
# ===================================================================

class TestCleaningPipeline:
    def test_returns_structured_dict(self, messy_df):
        result = cleaning_pipeline(messy_df)
        required = {
            "cleaned_df", "original_shape", "cleaned_shape",
            "nulls_handled", "duplicates_removed", "type_fixes",
            "columns_dropped", "outliers_detected", "outlier_method",
            "skew_report", "high_skew", "skew_kpi",
            "id_columns", "protected_columns", "profile_used", "logs",
        }
        assert required <= set(result.keys())

    def test_output_is_cleaner(self, messy_df):
        result = cleaning_pipeline(messy_df)
        assert result["cleaned_shape"][0] <= result["original_shape"][0]

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
        with pytest.raises(ScrubTypeError):
            cleaning_pipeline({"a": [1, 2, 3]})

    def test_rejects_invalid_step_names(self, messy_df):
        with pytest.raises(ScrubValueError, match="Invalid step"):
            cleaning_pipeline(messy_df, steps={"not_a_real_step"})

    def test_step_selection(self, messy_df):
        result = cleaning_pipeline(messy_df, steps={"duplicates"})
        assert result["outlier_method"] == "skipped"

    def test_does_not_mutate_input(self, messy_df):
        original_shape = messy_df.shape
        original_columns = messy_df.columns.tolist()
        cleaning_pipeline(messy_df)
        assert messy_df.shape == original_shape
        assert messy_df.columns.tolist() == original_columns

    def test_profile_auto(self, messy_df):
        result = cleaning_pipeline(messy_df, profile="auto")
        assert result["profile_used"] == "auto"

    def test_profile_transactional(self, messy_df):
        result = cleaning_pipeline(messy_df, profile="transactional")
        assert result["profile_used"] == "transactional"

    def test_profile_survey(self, survey_df):
        result = cleaning_pipeline(survey_df, profile="survey")
        assert result["profile_used"] == "survey"
        # Survey profile should preserve skip-logic missingness
        # q4 should still have nulls where q3=0
        cleaned = result["cleaned_df"]
        assert cleaned.isnull().sum().sum() > 0 or "q4" not in str(cleaned.columns)

    def test_explicit_params_override_profile(self, messy_df):
        result = cleaning_pipeline(
            messy_df,
            profile="survey",
            missing_strategy="median",
            outlier_mode="flag",
        )
        assert result["profile_used"] == "survey"

    def test_id_columns_detected(self, id_df):
        result = cleaning_pipeline(id_df)
        assert "student_id" in result["id_columns"]

    def test_id_columns_excluded_from_outliers(self, id_df):
        result = cleaning_pipeline(id_df)
        # student_id should NOT appear in outlier report
        assert "student_id" not in result["outliers_detected"]


# ===================================================================
# 2. handle_missing_values
# ===================================================================

class TestHandleMissingValues:
    def test_median_strategy(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="median")
        assert result[["num_a", "num_b"]].isnull().sum().sum() == 0

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

    def test_skip_strategy(self, missing_df):
        result = handle_missing_values(missing_df.copy(), strategy="skip")
        assert result.isnull().sum().sum() == missing_df.isnull().sum().sum()

    def test_invalid_strategy_raises(self, missing_df):
        with pytest.raises(ScrubValueError):
            handle_missing_values(missing_df.copy(), strategy="invalid")

    def test_exclude_cols(self, missing_df):
        result = handle_missing_values(
            missing_df.copy(), strategy="median", exclude_cols={"num_a"}
        )
        # num_a should still have nulls
        assert result["num_a"].isnull().sum() > 0
        # num_b should be filled
        assert result["num_b"].isnull().sum() == 0


# ===================================================================
# 3. detect_and_remove_outliers — flag mode + exclude
# ===================================================================

class TestDetectAndRemoveOutliers:
    def test_iqr_removes_outliers(self, numeric_df):
        numeric_df.loc[0, "normal"] = 10000
        cleaned, method, report, kpi = detect_and_remove_outliers(
            numeric_df, method="IQR"
        )
        assert cleaned.shape[0] < numeric_df.shape[0]

    def test_zscore_removes_outliers(self, numeric_df):
        numeric_df.loc[0, "normal"] = 10000
        cleaned, method, report, kpi = detect_and_remove_outliers(
            numeric_df, method="Z-score"
        )
        assert cleaned.shape[0] < numeric_df.shape[0]

    def test_flag_mode(self, numeric_df):
        numeric_df.loc[0, "normal"] = 10000
        result, method, report, kpi = detect_and_remove_outliers(
            numeric_df, method="IQR", mode="flag"
        )
        # Rows should NOT be removed
        assert result.shape[0] == numeric_df.shape[0]
        # _is_outlier column should exist
        assert "_is_outlier" in result.columns
        assert result["_is_outlier"].sum() > 0

    def test_skip_mode(self, numeric_df):
        result, method, _, _ = detect_and_remove_outliers(
            numeric_df, mode="skip"
        )
        assert method == "skipped"
        assert result.shape == numeric_df.shape

    def test_exclude_cols(self, numeric_df):
        numeric_df.loc[0, "normal"] = 10000
        _, _, report, _ = detect_and_remove_outliers(
            numeric_df, method="IQR", exclude_cols={"normal"}
        )
        assert "normal" not in report

    def test_invalid_method_raises(self, numeric_df):
        with pytest.raises(ScrubValueError):
            detect_and_remove_outliers(numeric_df, method="Invalid")

    def test_no_numeric_columns_skips(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        _, method, _, _ = detect_and_remove_outliers(df)
        assert method == "skipped"


# ===================================================================
# 4. convert_column_type — with exclude
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
    
    def test_handles_mixed_types(self):
        df = pd.DataFrame({"mixed": [1,2,"3"]})
        result = convert_column_type(df)
        assert pd.api.types.is_numeric_dtype(result["mixed"])
        assert result["mixed"].iloc[2] == 3

    def test_handles_commas_in_numbers(self):
        df = pd.DataFrame({"amount": ["1,000", "2,500", "3,750", " 343 ", "$540"]})
        result = convert_column_type(df)
        assert pd.api.types.is_numeric_dtype(result["amount"])

    def test_exclude_cols_skips_conversion(self):
        df = pd.DataFrame({"id": ["1", "2", "3"], "val": ["4", "5", "6"]})
        result = convert_column_type(df, exclude_cols={"id"})
        assert not pd.api.types.is_numeric_dtype(result["id"])
        assert pd.api.types.is_numeric_dtype(result["val"])


# ===================================================================
# 5. remove_duplicates
# ===================================================================

class TestRemoveDuplicates:
    def test_removes_exact_row_duplicates(self, duplicate_df):
        result, _ = remove_duplicates(duplicate_df)
        assert result.shape[0] == 3

    def test_handles_duplicate_columns(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        result, dup_cols = remove_duplicates(df)
        assert "a" in dup_cols

    def test_no_change_on_unique_data(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, _ = remove_duplicates(df)
        assert result.shape == df.shape


# ===================================================================
# 6. New cleaning steps
# ===================================================================

class TestDropHighNullColumns:
    def test_drops_above_threshold(self):
        df = pd.DataFrame({
            "mostly_null": [None] * 80 + [1.0] * 20,
            "fine": range(100),
        })
        result = drop_high_null_columns(df, threshold=0.7)
        assert "mostly_null" not in result.columns
        assert "fine" in result.columns

    def test_protects_specified_columns(self):
        df = pd.DataFrame({
            "skip_q": [None] * 80 + [1.0] * 20,
            "fine": range(100),
        })
        result = drop_high_null_columns(
            df, threshold=0.7, protect_cols={"skip_q"}
        )
        assert "skip_q" in result.columns

    def test_no_drop_below_threshold(self):
        df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
        result = drop_high_null_columns(df, threshold=0.7)
        assert set(result.columns) == {"a", "b"}


class TestDropConstantColumns:
    def test_drops_constant(self):
        df = pd.DataFrame({"const": [42] * 100, "varied": range(100)})
        result = drop_constant_columns(df)
        assert "const" not in result.columns
        assert "varied" in result.columns

    def test_keeps_non_constant(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = drop_constant_columns(df)
        assert set(result.columns) == {"a", "b"}


class TestNormalizeEncoding:
    def test_normalizes_curly_quotes(self):
        df = pd.DataFrame({"text": ["\u201cHello\u201d", "it\u2019s fine"]})
        result = normalize_encoding_step(df)
        assert result["text"].iloc[0] == '"Hello"'
        assert result["text"].iloc[1] == "it's fine"

    def test_normalizes_em_dash(self):
        df = pd.DataFrame({"text": ["A\u2014B"]})
        result = normalize_encoding_step(df)
        assert result["text"].iloc[0] == "A-B"


class TestCleanUnnamedColumns:
    def test_drops_all_null_unnamed(self):
        df = pd.DataFrame({"Unnamed: 0": [None, None], "real": [1, 2]})
        result = clean_unnamed_columns(df)
        assert "Unnamed: 0" not in result.columns

    def test_renames_non_null_unnamed(self):
        df = pd.DataFrame({"Unnamed: 0": [1, 2], "real": [3, 4]})
        result = clean_unnamed_columns(df)
        assert "renamed_1" in result.columns


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