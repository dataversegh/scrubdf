"""scrubdf - Scrub, clean, and explore DataFrames.

Automated data cleaning and EDA pipeline for pandas DataFrames.

Quick start::

    import pandas as pd
    from scrubdf.io import read_file
    from scrubdf.cleaning import cleaning_pipeline

    df = read_file("messy_data.csv")
    result = cleaning_pipeline(df)

    clean_df = result["cleaned_df"]
    print(f"Removed {result['duplicates_removed']} duplicates")
    print(f"Handled {result['nulls_handled']} null values")
"""

__version__ = "0.1.0"

# Public API — importable directly from `scrubdf`
from scrubdf.cleaning import cleaning_pipeline, ALL_STEPS
from scrubdf.io import read_file, list_sheets, SUPPORTED_EXTENSIONS
from scrubdf.profiles import get_profile, DataProfile, PROFILE_AUTO, PROFILE_SURVEY, PROFILE_TRANSACTIONAL
from scrubdf.utils import ScrubError, ScrubFileError, ScrubTypeError, ScrubValueError

__all__ = [
    # Core pipeline
    "cleaning_pipeline",
    "ALL_STEPS",
    # File I/O
    "read_file",
    "list_sheets",
    "SUPPORTED_EXTENSIONS",
    # Profiles
    "get_profile",
    "DataProfile",
    "PROFILE_AUTO",
    "PROFILE_SURVEY",
    "PROFILE_TRANSACTIONAL",
    # Exceptions
    "ScrubError",
    "ScrubFileError",
    "ScrubTypeError",
    "ScrubValueError",
]