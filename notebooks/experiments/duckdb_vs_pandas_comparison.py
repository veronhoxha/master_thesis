### IMPORTS ###
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################

"""

Compare DuckDB validation results with pandas row count summary.

This script helps confirm whether both pipelines retain the same number of rows
per model/domain/shot combination.

"""

BASE_DIR = Path(__file__).resolve().parents[2]
DUCKDB_JSON_PATH = (
    BASE_DIR / "analysis" / "duckdb_validation" / "duckdb_validation_results.json"
)
PANDAS_SUMMARY_PATH = (
    BASE_DIR / "analysis" / "basic_details" / "raw_finals_comprehensive_analysis.csv"
)


def load_duckdb_results(json_path: Path) -> Dict[str, dict]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records: Dict[str, dict] = {}
    for key, entry in payload.items():
        results = entry.get("results", [])
        if not results:
            continue
        first = results[0]
        records[key] = {
            "duckdb_row_count": first.get("row_count"),
            "duckdb_column_count": first.get("column_count"),
            "duckdb_valid": first.get("valid"),
            "duckdb_sql_test_passed": first.get("sql_test_passed"),
            "duckdb_error": first.get("error"),
        }
    return records


def load_pandas_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["key"] = df["model_run"] + "_" + df["domain"] + "_" + df["shot_type"]
    return df


def merge_results(
    duckdb_records: Dict[str, dict], pandas_df: pd.DataFrame
) -> pd.DataFrame:
    """Align DuckDB and pandas metrics and compute row differences."""
    duckdf = (
        pd.DataFrame.from_dict(duckdb_records, orient="index")
        .rename_axis("key")
        .reset_index()
    )

    merged = pandas_df.merge(duckdf, on="key", how="outer", indicator=True)
    merged["row_diff_duck_vs_pandas"] = (
        merged["duckdb_row_count"] - merged["pandas_readable_rows"]
    )
    merged["total_minus_duck_row_diff"] = (
        merged["total_data_rows"] - merged["duckdb_row_count"]
    )
    return merged


def format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "model_run",
        "domain",
        "shot_type",
        "duckdb_row_count",
        "pandas_readable_rows",
        "row_diff_duck_vs_pandas",
        "pandas_skipped_rows",
        "total_minus_duck_row_diff",
        "duckdb_valid",
        "duckdb_sql_test_passed",
        "_merge",
    ]
    return df.loc[:, [col for col in columns if col in df.columns]]


def print_report(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    total = len(df)
    mismatches = df[df["row_diff_duck_vs_pandas"].fillna(0) != 0]
    missing_duck = df[df["_merge"] == "left_only"]
    missing_pandas = df[df["_merge"] == "right_only"]

    print(f"Compared {total} dataset.")
    print(f"- DuckDB vs pandas row mismatches: {len(mismatches)}")
    print(f"- Entries missing DuckDB results: {len(missing_duck)}")
    print(f"- Entries missing pandas summary: {len(missing_pandas)}")

    if not mismatches.empty:
        print("\nTop mismatches:")
        display_df = (
            mismatches.assign(
                abs_diff=lambda d: d["row_diff_duck_vs_pandas"].abs()
            )
            .sort_values("abs_diff", ascending=False)
            .drop(columns="abs_diff")
        )
        if max_rows is not None:
            display_df = display_df.head(max_rows)
        print(display_df.to_string(index=False))
    else:
        print("\nDuckDB and pandas row counts match for all entries.")


def write_output(df: pd.DataFrame, output_path: Optional[Path]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main(argv: Optional[Iterable[str]] = None) -> None:
    duck_records = load_duckdb_results(DUCKDB_JSON_PATH)
    pandas_df = load_pandas_summary(PANDAS_SUMMARY_PATH)
    merged = merge_results(duck_records, pandas_df)
    summary = format_summary_table(merged)

    print_report(summary, max_rows=10)
    write_output(summary, BASE_DIR / "analysis" / "duckdb_validation" / "duckdb_vs_pandas_comparison.csv")


if __name__ == "__main__":
    main()