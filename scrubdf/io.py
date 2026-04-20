"""File ingestion — read any supported file format into a pandas DataFrame.

Supported formats:
    .csv, .tsv, .xlsx, .xls, .xlsm, .dta, .sas7bdat, .xpt,
    .sav, .parquet, .json, .feather, .orc, .pkl/.pickle

Usage::

    from scrubdf.io import read_file

    df = read_file("survey_results.xlsx")
    df = read_file("big_dataset.sas7bdat")
    df = read_file("export.dta")

    # From an uploaded file object (FastAPI / Streamlit)
    df = read_file(uploaded_file, filename="data.csv")

    # Excel with a specific sheet
    df = read_file("multi_sheet.xlsx", sheet_name="Wave 2")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, BinaryIO, Union

import pandas as pd

from scrubdf.utils import ScrubFileError, ScrubValueError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum file size in bytes (default 500 MB).
MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024

_READERS: dict[str, str] = {
    ".csv": "_read_csv",
    ".tsv": "_read_tsv",
    ".xlsx": "_read_excel",
    ".xls": "_read_excel",
    ".xlsm": "_read_excel",
    ".dta": "_read_stata",
    ".sas7bdat": "_read_sas",
    ".xpt": "_read_xport",
    ".sav": "_read_spss",
    ".parquet": "_read_parquet",
    ".json": "_read_json",
    ".feather": "_read_feather",
    ".orc": "_read_orc",
    ".pkl": "_read_pickle",
    ".pickle": "_read_pickle",
}

SUPPORTED_EXTENSIONS: list[str] = sorted(_READERS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_file(
    source: Union[str, Path, BinaryIO],
    *,
    filename: str | None = None,
    sheet_name: Union[str, int, None] = 0,
    encoding: str | None = None,
    separator: str | None = None,
    max_file_size: int = MAX_FILE_SIZE_BYTES,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a file into a pandas DataFrame.

    Parameters
    ----------
    source : str, Path, or file-like object
        File path or an in-memory file object (e.g. from FastAPI's
        ``UploadFile.file`` or Streamlit's ``st.file_uploader``).
    filename : str, optional
        Required when *source* is a file-like object so we can detect
        the format from the extension.  Ignored when *source* is a path.
    sheet_name : str, int, or None
        For Excel files only.  ``0`` = first sheet (default),
        ``None`` = all sheets concatenated, or pass a sheet name string.
    encoding : str, optional
        Text encoding for CSV/TSV.  If ``None``, tries ``utf-8`` first,
        then falls back to ``latin-1``, then ``cp1252``.
    separator : str, optional
        Column separator for CSV/TSV.  If ``None``, auto-detected.
    max_file_size : int
        Maximum file size in bytes (default 500 MB).
    **kwargs
        Extra keyword arguments forwarded to the underlying pandas reader.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ScrubValueError
        If the file extension is unrecognised or *filename* is missing
        for file-like objects.
    ScrubFileError
        If the file cannot be read, is empty, is corrupted, or exceeds
        the size limit.
    FileNotFoundError
        If a path is given and the file doesn't exist.
    """
    ext = _resolve_extension(source, filename)
    reader_name = _READERS.get(ext)

    if reader_name is None:
        raise ScrubValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Path-based checks
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        file_size = path.stat().st_size
        if file_size == 0:
            raise ScrubFileError(f"File is empty (0 bytes): {path}")
        if file_size > max_file_size:
            size_mb = file_size / (1024 * 1024)
            limit_mb = max_file_size / (1024 * 1024)
            raise ScrubFileError(
                f"File is {size_mb:.0f} MB, which exceeds the "
                f"{limit_mb:.0f} MB limit. Increase max_file_size if needed."
            )

    reader_fn = globals()[reader_name]

    try:
        df = reader_fn(
            source, encoding=encoding, separator=separator,
            sheet_name=sheet_name, **kwargs,
        )
    except (ScrubFileError, ScrubValueError, FileNotFoundError, ImportError):
        raise
    except Exception as e:
        raise ScrubFileError(
            f"Failed to read '{filename or source}' as {ext} file: {e}"
        ) from e

    # Post-read validation
    if not isinstance(df, pd.DataFrame):
        raise ScrubFileError(
            f"Reader returned {type(df).__name__} instead of DataFrame"
        )
    if df.empty:
        raise ScrubFileError(
            f"File '{filename or source}' produced an empty DataFrame. "
            f"Check that the file contains data and the correct sheet/encoding."
        )

    return df


def list_sheets(source: Union[str, Path, BinaryIO]) -> list[str]:
    """Return sheet names for an Excel file.

    Useful when the user needs to pick a sheet before calling ``read_file``.

    Parameters
    ----------
    source : str, Path, or file-like object
        Path to an Excel file.

    Returns
    -------
    list of str
        Sheet names.

    Raises
    ------
    ScrubFileError
        If the file cannot be opened as Excel.
    """
    try:
        xls = pd.ExcelFile(source)
        return xls.sheet_names
    except Exception as e:
        raise ScrubFileError(f"Could not read Excel file: {e}") from e


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------

def _read_csv(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    sep = separator or ","
    return _read_delimited(source, sep=sep, encoding=encoding, **kwargs)


def _read_tsv(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    sep = separator or "\t"
    return _read_delimited(source, sep=sep, encoding=encoding, **kwargs)


def _read_delimited(source, *, sep, encoding=None, **kwargs):
    """Read CSV/TSV with encoding fallback."""
    encodings_to_try = [encoding] if encoding else ["utf-8", "latin-1", "cp1252"]

    last_error: Exception | None = None
    for enc in encodings_to_try:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, sep=sep, encoding=enc, **kwargs)
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except pd.errors.ParserError as e:
            raise ScrubFileError(
                f"CSV/TSV parsing failed (malformed rows, mismatched columns, "
                f"or wrong separator). Try specifying separator='\\t' or ','.\n"
                f"Parser error: {e}"
            ) from e

    raise ScrubFileError(
        f"Could not decode file with encodings {encodings_to_try}. "
        f"Last error: {last_error}"
    )


def _read_excel(source, *, encoding=None, separator=None, sheet_name=0, **kwargs):
    try:
        if sheet_name is None:
            all_sheets = pd.read_excel(source, sheet_name=None, **kwargs)
            frames = []
            for name, df in all_sheets.items():
                df = df.copy()
                df["_sheet_name"] = name
                frames.append(df)
            if not frames:
                raise ScrubFileError("Excel file contains no sheets")
            return pd.concat(frames, ignore_index=True)
        return pd.read_excel(source, sheet_name=sheet_name, **kwargs)
    except ScrubFileError:
        raise
    except Exception as e:
        if "password" in str(e).lower() or "encrypt" in str(e).lower():
            raise ScrubFileError(
                "Excel file appears to be password-protected. "
                "Please remove the password and try again."
            ) from e
        raise ScrubFileError(f"Could not read Excel file: {e}") from e


def _read_stata(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_stata(source, **kwargs)


def _read_sas(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    enc = encoding or "latin-1"
    return pd.read_sas(source, format="sas7bdat", encoding=enc, **kwargs)


def _read_xport(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_sas(source, format="xport", **kwargs)


def _read_spss(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    try:
        return pd.read_spss(source, **kwargs)
    except ImportError:
        raise ImportError(
            "Reading SPSS (.sav) files requires pyreadstat. "
            "Install it with: pip install pyreadstat"
        )


def _read_parquet(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_parquet(source, **kwargs)


def _read_json(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_json(source, **kwargs)


def _read_feather(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_feather(source, **kwargs)


def _read_orc(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_orc(source, **kwargs)


def _read_pickle(source, *, encoding=None, separator=None, sheet_name=None, **kwargs):
    return pd.read_pickle(source, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_extension(
    source: Union[str, Path, BinaryIO],
    filename: str | None,
) -> str:
    """Extract and normalise the file extension."""
    if isinstance(source, (str, Path)):
        ext = Path(source).suffix.lower()
    elif filename:
        ext = Path(filename).suffix.lower()
    elif hasattr(source, "name"):
        ext = Path(source.name).suffix.lower()
    else:
        raise ScrubValueError(
            "Cannot determine file format. Pass filename='data.csv' "
            "when using a file-like object."
        )

    if not ext:
        raise ScrubValueError(
            "File has no extension. Pass filename='data.csv' to specify format."
        )

    return ext
