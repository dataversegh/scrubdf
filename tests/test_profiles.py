"""Tests for scrubdf.profiles and column classification."""

import numpy as np
import pandas as pd
import pytest

from scrubdf.profiles import (
    DataProfile,
    get_profile,
    detect_ordinal_columns,
    detect_coded_categoricals,
    detect_skip_logic_columns,
    PROFILE_AUTO,
    PROFILE_SURVEY,
    PROFILE_TRANSACTIONAL,
)
from scrubdf.utils import detect_id_columns


# ===================================================================
# Profile lookup
# ===================================================================

class TestGetProfile:
    def test_auto(self):
        p = get_profile("auto")
        assert p.name == "auto"

    def test_survey(self):
        p = get_profile("survey")
        assert p.name == "survey"
        assert p.protect_ordinal_columns is True
        assert p.respect_skip_logic is True

    def test_transactional(self):
        p = get_profile("transactional")
        assert p.name == "transactional"
        assert p.drop_constant_columns is True

    def test_case_insensitive(self):
        p = get_profile("Survey")
        assert p.name == "survey"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")


# ===================================================================
# ID detection
# ===================================================================

class TestDetectIdColumns:
    def test_detects_by_name(self):
        df = pd.DataFrame({"student_id": [1, 2, 3], "age": [20, 25, 30]})
        assert "student_id" in detect_id_columns(df)

    def test_detects_suffix_id(self):
        df = pd.DataFrame({"user_id": [1, 2, 3], "score": [80, 90, 70]})
        assert "user_id" in detect_id_columns(df)

    def test_detects_respondent(self):
        df = pd.DataFrame({"respondent_code": ["R1", "R2"], "q1": [3, 4]})
        assert "respondent_code" in detect_id_columns(df)

    def test_detects_monotonic_unique(self):
        df = pd.DataFrame({
            "record_num": range(100),
            "value": np.random.randn(100),
        })
        ids = detect_id_columns(df)
        assert "record_num" in ids

    def test_does_not_flag_normal_numeric(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "income": np.random.normal(50000, 10000, 100),
            "age": np.random.randint(18, 65, 100),
        })
        ids = detect_id_columns(df)
        assert "income" not in ids
        assert "age" not in ids

    def test_empty_df(self):
        df = pd.DataFrame({"id": pd.Series(dtype="int64")})
        assert detect_id_columns(df) == ["id"]  # name-based still works


# ===================================================================
# Ordinal detection
# ===================================================================

class TestDetectOrdinalColumns:
    def test_detects_likert_scale(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "satisfaction": np.random.choice([1, 2, 3, 4, 5], 200),
            "income": np.random.normal(50000, 10000, 200),
        })
        ordinals = detect_ordinal_columns(df)
        assert "satisfaction" in ordinals
        assert "income" not in ordinals

    def test_ignores_wide_range(self):
        df = pd.DataFrame({"age": np.random.randint(18, 65, 200)})
        ordinals = detect_ordinal_columns(df)
        assert "age" not in ordinals

    def test_detects_binary(self):
        df = pd.DataFrame({"yes_no": np.random.choice([0, 1], 200)})
        ordinals = detect_ordinal_columns(df, max_unique=10)
        assert "yes_no" in ordinals


# ===================================================================
# Coded categorical detection
# ===================================================================

class TestDetectCodedCategoricals:
    def test_detects_coded_gender(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "gender_code": np.random.choice([1, 2, 3], 200),
            "income": np.random.normal(50000, 10000, 200),
        })
        coded = detect_coded_categoricals(df)
        assert "gender_code" in coded
        assert "income" not in coded

    def test_ignores_high_cardinality(self):
        df = pd.DataFrame({"varied": range(200)})
        assert detect_coded_categoricals(df) == []


# ===================================================================
# Skip logic detection
# ===================================================================

class TestDetectSkipLogicColumns:
    def test_detects_conditional_nulls(self, survey_df):
        skip_cols = detect_skip_logic_columns(survey_df)
        # q4_product_rating has nulls where q3_used_product=0
        assert "q4_product_rating" in skip_cols

    def test_no_skip_logic_in_clean_data(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
        })
        assert detect_skip_logic_columns(df) == []


# ===================================================================
# Profile defaults
# ===================================================================

class TestProfileDefaults:
    def test_survey_skips_outliers(self):
        assert PROFILE_SURVEY.outlier_mode == "skip"

    def test_survey_uses_mode_imputation(self):
        assert PROFILE_SURVEY.missing_strategy == "mode"

    def test_transactional_drops_high_null(self):
        assert PROFILE_TRANSACTIONAL.drop_high_null_threshold == 0.7

    def test_auto_protects_ids(self):
        assert PROFILE_AUTO.protect_id_columns is True

    def test_profiles_are_frozen(self):
        with pytest.raises(AttributeError):
            PROFILE_AUTO.name = "modified"