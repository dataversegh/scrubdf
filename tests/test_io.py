"""Tests for scrubdf.io — file reading, error handling, edge cases."""

import pandas as pd
import pytest

from scrubdf.io import read_file, SUPPORTED_EXTENSIONS, list_sheets
from scrubdf.utils import ScrubFileError, ScrubValueError


class TestReadFile:
    def test_reads_csv(self, tmp_csv):
        df = read_file(tmp_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)

    def test_reads_csv_string_path(self, tmp_csv):
        df = read_file(str(tmp_csv))
        assert df.shape == (3, 2)

    def test_reads_file_object_with_filename(self, tmp_csv):
        with open(tmp_csv, "rb") as f:
            df = read_file(f, filename="test.csv")
        assert df.shape == (3, 2)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_file("/nonexistent/path/data.csv")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("some data")
        with pytest.raises(ScrubValueError, match="Unsupported file extension"):
            read_file(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ScrubFileError, match="empty"):
            read_file(p)

    def test_file_object_without_filename_raises(self):
        import io
        buf = io.BytesIO(b"a,b\n1,2")
        # Remove name attribute if present
        if hasattr(buf, "name"):
            del buf.name
        with pytest.raises(ScrubValueError, match="Cannot determine"):
            read_file(buf)

    def test_malformed_csv_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        # Write a file that will produce an empty DataFrame
        p.write_text("a,b,c\n")
        with pytest.raises(ScrubFileError):
            read_file(p)

    def test_tsv_reads_correctly(self, tmp_path):
        p = tmp_path / "data.tsv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(
            p, sep="\t", index=False
        )
        df = read_file(p)
        assert df.shape == (2, 2)


class TestSupportedExtensions:
    def test_csv_in_supported(self):
        assert ".csv" in SUPPORTED_EXTENSIONS

    def test_xlsx_in_supported(self):
        assert ".xlsx" in SUPPORTED_EXTENSIONS

    def test_dta_in_supported(self):
        assert ".dta" in SUPPORTED_EXTENSIONS

    def test_sas_in_supported(self):
        assert ".sas7bdat" in SUPPORTED_EXTENSIONS


class TestListSheets:
    def test_list_sheets_on_non_excel_raises(self, tmp_csv):
        with pytest.raises(ScrubFileError):
            list_sheets(tmp_csv)
