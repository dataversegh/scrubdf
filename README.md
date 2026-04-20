# scrubdf

**Scrub, clean, and explore DataFrames.**

`scrubdf` is an automated data cleaning and exploratory data analysis pipeline for pandas DataFrames. Feed it messy data in any format, get back a clean DataFrame and a structured report of everything that changed.

## Install

```bash
pip install scrubdf

# With Isolation Forest outlier detection
pip install scrubdf[ml]

# With dev tools (pytest)
pip install scrubdf[dev]
```

## Quick start

```python
from scrubdf import read_file, cleaning_pipeline

# Read any file format — CSV, Excel, Stata, SAS, SPSS, Parquet, JSON...
df = read_file("messy_data.csv")

# Run the full cleaning pipeline
result = cleaning_pipeline(df)

clean_df = result["cleaned_df"]
print(f"Shape: {result['original_shape']} → {result['cleaned_shape']}")
print(f"Nulls handled: {result['nulls_handled']}")
print(f"Duplicates removed: {result['duplicates_removed']}")
print(f"Outlier method: {result['outlier_method']}")

# Review what happened
for log_entry in result["logs"]:
    print(log_entry)
```

## Choose what runs

```python
result = cleaning_pipeline(
    df,
    steps={"duplicates", "missing_values", "outliers"},
    missing_strategy="mean",
    outlier_method="IQR",
)
```

Available steps: `unnamed_columns`, `column_names`, `whitespace`, `type_conversion`, `standardise_values`, `empty_rows_cols`, `duplicates`, `missing_values`, `date_parsing`, `outliers`.

## Read any file format

```python
from scrubdf import read_file

df = read_file("data.xlsx", sheet_name="Sheet2")
df = read_file("survey.sas7bdat")
df = read_file("export.dta")

# From a file upload (FastAPI / Streamlit)
df = read_file(uploaded_file.file, filename="data.csv")
```

Supported: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.xlsm`, `.dta`, `.sas7bdat`, `.xpt`, `.sav`, `.parquet`, `.json`, `.feather`, `.orc`, `.pkl`

## Use individual functions

```python
from scrubdf.cleaning import handle_missing_values, detect_and_remove_outliers
from scrubdf.eda import skewness_check, corr_matrix, missing_eda
from scrubdf.kpi import kpi_card, corr_pair

# Clean
df = handle_missing_values(df, strategy="median")
df, method, report, kpi = detect_and_remove_outliers(df, method="IQR")

# Analyse
skew_df, high_skew, skew_kpi = skewness_check(df)
correlations = corr_matrix(df)

# Compute KPIs (returns dicts, not HTML)
card = kpi_card("Total Rows", len(df))
# {"label": "Total Rows", "raw_value": 12847, "display_value": "12.8K"}
```

## Error handling

scrubdf uses typed exceptions so you can catch specific failures:

```python
from scrubdf import ScrubFileError, ScrubValueError, ScrubTypeError

try:
    df = read_file("data.xlsx")
    result = cleaning_pipeline(df)
except ScrubFileError as e:
    print(f"File problem: {e}")
except ScrubValueError as e:
    print(f"Data problem: {e}")
```

## Logging

All pipeline activity is logged via Python's `logging` module under the `scrubdf` logger. Configure it however your infrastructure needs:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now pipeline logs appear in your console / Cloud Logging / etc.
result = cleaning_pipeline(df)
```

## Pipeline output

`cleaning_pipeline()` returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `cleaned_df` | DataFrame | The cleaned data |
| `original_shape` | tuple | (rows, cols) before |
| `cleaned_shape` | tuple | (rows, cols) after |
| `nulls_handled` | int | Null values filled or dropped |
| `duplicates_removed` | int | Duplicate rows removed |
| `type_fixes` | int | Columns whose dtype changed |
| `columns_dropped` | list | Columns removed |
| `outliers_detected` | dict | {column: count} |
| `outlier_method` | str | Method used |
| `skew_report` | DataFrame | Skewness per column |
| `high_skew` | list | Highly skewed column descriptions |
| `skew_kpi` | dict | Top skewed columns |
| `logs` | list | Timestamped log messages |

## Development

```bash
git clone https://github.com/dataversegh/scrubdf.git
cd scrubdf
pip install -e ".[dev]"
pytest
```

## License

MIT
