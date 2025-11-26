"""
Dataset quality checks for comparing LLM-generated data against real reference datasets.

Runn `python notebooks/experiments/dataset_quality_checks.py`
This script processes every cleaned generated dataset detected under `data/generated/<model-run>/<domain>/<shot>/csv/`.
Real datasets are loaded from `data/preprocessed/`.
"""

### IMPORTS ###
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


###########################


def find_project_root() -> Path:
    """Locate the repository root by walking upwards until README.md is found."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
QUALITY_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "quality_checks"
QUALITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_PATH = PREPROCESSED_DIR / "column_type_ground_truth.json"


@dataclass
class DatasetConfig:
    domain: str
    real_path: Path
    primary_key: Optional[List[str]] = None
    date_columns: List[str] = field(default_factory=list)
    date_order_constraints: List[Tuple[str, str]] = field(default_factory=list)
    date_formats: Dict[str, str] = field(default_factory=dict)
    non_negative_columns: List[str] = field(default_factory=list)
    percent_columns: List[str] = field(default_factory=list)
    bounded_columns: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)


@dataclass
class RealProfile:
    simplified_dtypes: Dict[str, str]
    numeric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]]
    categorical_allowed: Dict[str, List[str]]
    date_ranges: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]


@dataclass
class GeneratedDatasetInfo:
    domain: str
    model: str
    shot: str
    path: Path
    run_id: str
    run_name: str


def load_ground_truth_types(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


GROUND_TRUTH_TYPES = load_ground_truth_types(GROUND_TRUTH_PATH)


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "hatecrime": DatasetConfig(
        domain="hatecrime",
        real_path=PREPROCESSED_DIR / "hate_crime_preprocessed.csv",
        primary_key=[
            "data_year",
            "state_abbr",
            "incident_date",
            "offense_name",
            "bias_desc",
        ],
        date_columns=["incident_date"],
        date_formats={"incident_date": "%Y-%m-%d"},
        non_negative_columns=[
            "adult_victim_count",
            "juvenile_victim_count",
            "total_offender_count",
            "adult_offender_count",
            "juvenile_offender_count",
            "victim_count",
            "total_individual_victims",
        ],
    ),
    "employment": DatasetConfig(
        domain="employment",
        real_path=PREPROCESSED_DIR / "uk_gender_pay_gap_data_2024_to_2025_preproccesed.csv",
        primary_key=["EmployerName", "DateSubmitted"],
        date_columns=["DateSubmitted"],
        date_formats={"DateSubmitted": "%Y-%m-%d %H:%M:%S"},
        non_negative_columns=[
            "MaleBonusPercent",
            "FemaleBonusPercent",
            "MaleLowerQuartile",
            "FemaleLowerQuartile",
            "MaleLowerMiddleQuartile",
            "FemaleLowerMiddleQuartile",
            "MaleUpperMiddleQuartile",
            "FemaleUpperMiddleQuartile",
            "MaleTopQuartile",
            "FemaleTopQuartile",
        ],
        percent_columns=[
            "DiffMeanHourlyPercent",
            "DiffMedianHourlyPercent",
            "DiffMeanBonusPercent",
            "DiffMedianBonusPercent",
            "MaleBonusPercent",
            "FemaleBonusPercent",
            "MaleLowerQuartile",
            "FemaleLowerQuartile",
            "MaleLowerMiddleQuartile",
            "FemaleLowerMiddleQuartile",
            "MaleUpperMiddleQuartile",
            "FemaleUpperMiddleQuartile",
            "MaleTopQuartile",
            "FemaleTopQuartile",
        ],
        bounded_columns={
            "DiffMeanHourlyPercent": (-100.0, 100.0),
            "DiffMedianHourlyPercent": (-100.0, 100.0),
            "DiffMeanBonusPercent": (-100.0, 100.0),
            "DiffMedianBonusPercent": (-100.0, 100.0),
        },
    ),
    "lending": DatasetConfig(
        domain="lending",
        real_path=PREPROCESSED_DIR / "year_2024_preprocessed.csv",
        primary_key=[
            "activity_year",
            "state_code",
            "county_code",
            "census_tract",
            "loan_purpose",
            "loan_amount",
        ],
        non_negative_columns=[
            "loan_amount",
            "loan_to_value_ratio",
            "interest_rate",
            "rate_spread",
            "total_loan_costs",
            "total_points_and_fees",
            "origination_charges",
            "discount_points",
            "lender_credits",
            "loan_term",
            "prepayment_penalty_term",
            "intro_rate_period",
            "property_value",
            "total_units",
            "multifamily_affordable_units",
            "income",
        ],
    ),
}


def simplify_dtype(dtype: Any) -> str:
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    return "object"


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_datetime(series: pd.Series, format: Optional[str] = None) -> pd.Series:
    if series.dtype == "datetime64[ns]" or series.dtype == "datetime64[ns, UTC]":
        return series
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series
    if format:
        parsed = pd.to_datetime(series, errors="coerce", format=format)
        missing_mask = parsed.isna() & series.notna()
        if missing_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fallback = pd.to_datetime(series[missing_mask], errors="coerce", utc=False)
            if not fallback.empty:
                parsed.loc[missing_mask] = fallback
        return parsed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce", utc=False)


def list_generated_datasets(config: DatasetConfig) -> List[GeneratedDatasetInfo]:
    if not GENERATED_DIR.exists():
        return []
    datasets: List[GeneratedDatasetInfo] = []
    for run_dir in sorted([p for p in GENERATED_DIR.iterdir() if p.is_dir()]):
        run_name = run_dir.name
        match = re.search(r"^(.*?)-run(\d+)$", run_name, flags=re.IGNORECASE)
        if match:
            base_model = match.group(1)
            run_id = f"run{match.group(2)}"
        else:
            base_model = run_name
            run_id = "run1"

        domain_dir = run_dir / config.domain
        if not domain_dir.exists():
            continue
        for shot_dir in sorted([p for p in domain_dir.iterdir() if p.is_dir()]):
            shot = shot_dir.name
            csv_dir = shot_dir / "csv"
            if not csv_dir.exists():
                continue
            clean_files = sorted(csv_dir.glob("*raw_finals_of_all_chunks*_clean.csv"))
            if not clean_files:
                continue
            for csv_path in clean_files:
                datasets.append(
                    GeneratedDatasetInfo(
                        domain=config.domain,
                        model=base_model,
                        shot=shot,
                        path=csv_path,
                        run_id=run_id,
                        run_name=run_name,
                    )
                )
    return datasets


def load_dataframe(path: Path) -> pd.DataFrame:
    readers = (
        {"encoding": "utf-8", "on_bad_lines": "skip", "low_memory": False},
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "latin-1", "on_bad_lines": "skip", "low_memory": False},
        {"encoding": "latin-1", "low_memory": False},
    )
    for kwargs in readers:
        try:
            return pd.read_csv(path, **kwargs) 
        except Exception:
            continue
    raise RuntimeError(f"Unable to read CSV file: {path}")


def build_real_profile(config: DatasetConfig, df: pd.DataFrame) -> RealProfile:
    domain_types = GROUND_TRUTH_TYPES.get(config.domain, {})
    numeric_cols = set(domain_types.get("numeric", []))
    categorical_cols = set(domain_types.get("categorical", []))

    simplified = {col: simplify_dtype(dtype) for col, dtype in df.dtypes.items()}

    numeric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    categorical_allowed: Dict[str, List[str]] = {}
    date_ranges: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}

    for col in df.columns:
        if col in config.date_columns:
            parsed = to_datetime(df[col], config.date_formats.get(col))
            if parsed.notna().any():
                date_ranges[col] = (parsed.min(), parsed.max())
            else:
                date_ranges[col] = (None, None)

    for col in numeric_cols:
        if col not in df.columns:
            continue
        numeric_series = to_numeric(df[col])
        if not numeric_series.dropna().empty:
            numeric_ranges[col] = (
                float(numeric_series.min(skipna=True)),
                float(numeric_series.max(skipna=True)),
            )
        else:
            numeric_ranges[col] = (None, None)

    for col in categorical_cols:
        if col not in df.columns:
            continue
        uniques = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
        )
        categorical_allowed[col] = sorted(map(str, uniques))

    return RealProfile(
        simplified_dtypes=simplified,
        numeric_ranges=numeric_ranges,
        categorical_allowed=categorical_allowed,
        date_ranges=date_ranges,
    )


def compute_schema_checks(
    real_profile: RealProfile,
    generated_df: pd.DataFrame,
) -> Dict[str, Any]:
    generated_types = {col: simplify_dtype(dtype) for col, dtype in generated_df.dtypes.items()}
    mismatches: List[Dict[str, Any]] = []
    matches = 0
    expected_columns = real_profile.simplified_dtypes

    for col, expected in expected_columns.items():
        observed = generated_types.get(col)
        if observed is None:
            mismatches.append(
                {
                    "column": col,
                    "issue": "missing_in_generated",
                    "expected": expected,
                    "observed": None,
                }
            )
            continue

        if expected == "numeric" and observed in {"numeric"}:
            matches += 1
        elif expected == "object" and observed == "object":
            matches += 1
        elif expected == "datetime" and observed == "datetime":
            matches += 1
        elif expected == "bool" and observed == "bool":
            matches += 1
        else:
            mismatches.append(
                {
                    "column": col,
                    "issue": "dtype_mismatch",
                    "expected": expected,
                    "observed": observed,
                }
            )

    extra_columns = [
        col for col in generated_types.keys() if col not in expected_columns
    ]
    total_expected = len(expected_columns)
    type_correctness_pct = (matches / total_expected * 100.0) if total_expected else 0.0

    return {
        "type_correctness_pct": type_correctness_pct,
        "mismatched_columns": mismatches,
        "extra_columns": extra_columns,
        "expected_column_count": total_expected,
        "generated_column_count": len(generated_types),
    }



def compute_numeric_validity(
    config: DatasetConfig,
    real_profile: RealProfile,
    generated_df: pd.DataFrame,
) -> Dict[str, Any]:
    results: Dict[str, Dict[str, Any]] = {}
    for col, (min_val, max_val) in real_profile.numeric_ranges.items():
        if col not in generated_df.columns:
            continue

        series = to_numeric(generated_df[col])
        if series.dropna().empty:
            continue

        lower = min_val
        upper = max_val
        total_rows = int(len(series))
        non_null_rows = int(series.notna().sum())

        if col in config.non_negative_columns:
            lower = max(lower or 0.0, 0.0)
        if col in config.percent_columns:
            lower = 0.0 if lower is None else max(lower, 0.0)
            upper = 100.0 if upper is None else min(upper, 100.0)
        if col in config.bounded_columns:
            manual_lower, manual_upper = config.bounded_columns[col]
            if manual_lower is not None:
                lower = manual_lower if lower is None else max(lower, manual_lower)
            if manual_upper is not None:
                upper = manual_upper if upper is not None else min(upper, manual_upper)

        mask = pd.Series(True, index=series.index, dtype=bool)
        if lower is not None and not math.isinf(lower):
            mask &= series.ge(lower) | series.isna()
        if upper is not None and not math.isinf(upper):
            mask &= series.le(upper) | series.isna()

        violations = int((~mask).sum())

        non_negative_violations = None
        non_negative_violations_pct = None
        if col in config.non_negative_columns:
            non_negative_violations = int((series < 0).sum())
            non_negative_violations_pct = (
                (non_negative_violations / non_null_rows * 100.0) if non_null_rows else None
            )

        results[col] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "violations": violations,
            "violations_pct": (violations / non_null_rows * 100.0) if non_null_rows else None,
            "rows_evaluated": non_null_rows,
            "non_negative_violations": non_negative_violations,
            "non_negative_violations_pct": non_negative_violations_pct,
        }

    return results



def compute_categorical_sanity(
    real_profile: RealProfile,
    generated_df: pd.DataFrame,
) -> Dict[str, Any]:
    results: Dict[str, Dict[str, Any]] = {}
    for col, allowed_values in real_profile.categorical_allowed.items():
        if col not in generated_df.columns:
            continue

        gen_series = generated_df[col].dropna().astype(str)
        total_values = int(gen_series.size)
        if gen_series.empty:
            continue

        allowed_set = set(allowed_values)
        unexpected = sorted(set(gen_series.unique()) - allowed_set)
        unexpected_counts = (
            gen_series[gen_series.isin(unexpected)].value_counts().head(20).to_dict()
            if unexpected
            else {}
        )
        unexpected_total = int(sum(unexpected_counts.values())) if unexpected_counts else 0

        results[col] = {
            "unique_generated": int(gen_series.nunique(dropna=True)),
            "allowed_unique_real": len(allowed_values),
            "unexpected_values_sample": unexpected[:20],
            "unexpected_value_counts": unexpected_counts,
            "unexpected_value_pct": (unexpected_total / total_values * 100.0) if total_values else None,
            "values_evaluated": total_values,
        }

    return results



def compute_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "overall_missing_pct": 0.0,
            "per_column_missing_pct": {},
        }
    per_column = df.isna().mean() * 100.0
    overall = df.isna().to_numpy().mean() * 100.0
    return {
        "overall_missing_pct": float(overall),
        "per_column_missing_pct": {col: float(val) for col, val in per_column.items()},
    }


def compute_duplicates(df: pd.DataFrame, primary_key: Optional[Sequence[str]]) -> Dict[str, Any]:
    row_count = int(len(df))
    duplicates_total = int(df.duplicated().sum())
    duplicate_pct = (duplicates_total / row_count * 100.0) if row_count else 0.0
    pk_duplicates = None
    pk_duplicate_pct = None
    if primary_key and all(col in df.columns for col in primary_key):
        pk_duplicates = int(df.duplicated(subset=list(primary_key)).sum())
        pk_duplicate_pct = (pk_duplicates / row_count * 100.0) if row_count else None
    return {
        "duplicate_rows": duplicates_total,
        "duplicate_rows_pct": duplicate_pct,
        "duplicate_rows_primary_key": pk_duplicates,
        "duplicate_rows_primary_key_pct": pk_duplicate_pct,
        "row_count": row_count,
    }


def compute_outliers(df: pd.DataFrame, numeric_columns: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    outliers: Dict[str, Dict[str, Any]] = {}
    for col in numeric_columns:
        if col not in df.columns:
            continue
        series = to_numeric(df[col]).dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(q1) or pd.isna(q3) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        outliers[col] = {
            "count": count,
            "count_pct": (count / len(series) * 100.0) if len(series) else None,
            "values_evaluated": int(len(series)),
        }
    return outliers



def compute_date_validation(
    config: DatasetConfig,
    real_profile: RealProfile,
    generated_df: pd.DataFrame,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for col in config.date_columns:
        if col not in generated_df.columns:
            continue
        parsed = to_datetime(generated_df[col], config.date_formats.get(col))
        invalid = (parsed.isna()) & generated_df[col].notna()
        min_date = parsed.min()
        max_date = parsed.max()

        real_min, real_max = real_profile.date_ranges.get(col, (None, None))

        results[col] = {
            "invalid_values": int(invalid.sum()),
            "invalid_values_pct": (
                int(invalid.sum()) / int(parsed.notna().sum()) * 100.0
                if parsed.notna().sum() > 0
                else None
            ),
            "min_generated": min_date.isoformat() if pd.notna(min_date) else None,
            "max_generated": max_date.isoformat() if pd.notna(max_date) else None,
            "min_real": real_min.isoformat() if real_min is not None and pd.notna(real_min) else None,
            "max_real": real_max.isoformat() if real_max is not None and pd.notna(real_max) else None,
            "values_evaluated": int(parsed.notna().sum()),
        }
    return results




def evaluate_row_constraints(
    config: DatasetConfig,
    generated_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    violations: Dict[str, Dict[str, Any]] = {}
    total_rows = int(len(generated_df)) or 1

    if config.domain == "hatecrime":
        if {"adult_victim_count", "juvenile_victim_count", "victim_count"}.issubset(generated_df.columns):
            adults = to_numeric(generated_df["adult_victim_count"]).fillna(0)
            juveniles = to_numeric(generated_df["juvenile_victim_count"]).fillna(0)
            total = to_numeric(generated_df["victim_count"])
            mask = (adults + juveniles) <= total.fillna(np.inf)
            count = int((~mask).sum())
            violations["victim_count_covers_age_groups"] = {
                "violations": count,
                "violations_pct": count / total_rows * 100.0,
            }

        if {"total_offender_count", "adult_offender_count", "juvenile_offender_count"}.issubset(generated_df.columns):
            total_off = to_numeric(generated_df["total_offender_count"])
            component = (
                to_numeric(generated_df["adult_offender_count"]).fillna(0)
                + to_numeric(generated_df["juvenile_offender_count"]).fillna(0)
            )
            mask = component <= total_off.fillna(np.inf)
            count = int((~mask).sum())
            violations["offender_count_covers_age_groups"] = {
                "violations": count,
                "violations_pct": count / total_rows * 100.0,
            }

    if config.domain == "employment":
        quartile_pairs = [
            ("MaleLowerQuartile", "FemaleLowerQuartile"),
            ("MaleLowerMiddleQuartile", "FemaleLowerMiddleQuartile"),
            ("MaleUpperMiddleQuartile", "FemaleUpperMiddleQuartile"),
            ("MaleTopQuartile", "FemaleTopQuartile"),
        ]
        tolerance = 1.0
        for male_col, female_col in quartile_pairs:
            if {male_col, female_col}.issubset(generated_df.columns):
                male = to_numeric(generated_df[male_col])
                female = to_numeric(generated_df[female_col])
                total = male + female
                mask = total.sub(100).abs() <= tolerance
                count = int((~mask.fillna(True)).sum())
                violations[f"{male_col}_{female_col}_sum_to_100"] = {
                    "violations": count,
                    "violations_pct": count / total_rows * 100.0,
                }

    if config.domain == "lending":
        if "loan_amount" in generated_df.columns and "property_value" in generated_df.columns:
            loan_amount = to_numeric(generated_df["loan_amount"])
            property_value = to_numeric(generated_df["property_value"])
            mask = (loan_amount <= property_value) | property_value.isna()
            count = int((~mask.fillna(True)).sum())
            violations["loan_amount_not_greater_than_property_value"] = {
                "violations": count,
                "violations_pct": count / total_rows * 100.0,
            }

    for start_col, end_col in config.date_order_constraints:
        if {start_col, end_col}.issubset(generated_df.columns):
            start = to_datetime(generated_df[start_col], config.date_formats.get(start_col))
            end = to_datetime(generated_df[end_col], config.date_formats.get(end_col))
            mask = start <= end
            count = int((~mask.fillna(True)).sum())
            violations[f"{start_col}_le_{end_col}"] = {
                "violations": count,
                "violations_pct": count / total_rows * 100.0,
            }

    return violations



def compute_numeric_distribution_similarity(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    numeric_cols: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for col in numeric_cols:
        if col not in real_df.columns or col not in gen_df.columns:
            continue

        real = pd.to_numeric(real_df[col], errors="coerce").dropna()
        gen = pd.to_numeric(gen_df[col], errors="coerce").dropna()
        if real.empty or gen.empty:
            continue

        ks = ks_2samp(real, gen).statistic
        mean_diff = abs(real.mean() - gen.mean())
        std_diff = abs(real.std(ddof=1) - gen.std(ddof=1))

        results[col] = {
            "ks_distance": float(ks),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
        }

    return results


def compute_categorical_distribution_similarity(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    categorical_cols: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for col in categorical_cols:
        if col not in real_df.columns or col not in gen_df.columns:
            continue

        real_counts = real_df[col].dropna().astype(str).value_counts(normalize=True)
        gen_counts = gen_df[col].dropna().astype(str).value_counts(normalize=True)
        if real_counts.empty or gen_counts.empty:
            continue

        all_cats = sorted(set(real_counts.index) | set(gen_counts.index))
        real_vec = real_counts.reindex(all_cats, fill_value=0.0)
        gen_vec = gen_counts.reindex(all_cats, fill_value=0.0)

        jsd = float(jensenshannon(real_vec, gen_vec))

        results[col] = {
            "js_divergence": jsd,
        }

    return results


def compute_text_quality(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    text_cols: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for col in text_cols:
        if col not in real_df.columns or col not in gen_df.columns:
            continue

        real = real_df[col].dropna().astype(str)
        gen = gen_df[col].dropna().astype(str)

        if real.empty and gen.empty:
            continue

        avg_len_real = float(real.str.len().mean()) if not real.empty else float("nan")
        avg_len_gen = float(gen.str.len().mean()) if not gen.empty else float("nan")
        empty_pct_gen = float((gen == "").mean() * 100.0) if not gen.empty else float("nan")

        results[col] = {
            "avg_length_real": avg_len_real,
            "avg_length_generated": avg_len_gen,
            "empty_pct_generated": empty_pct_gen,
        }

    return results



def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not arr:
        return None
    return float(np.mean(arr))


def quality_checks_for_dataset(
    config: DatasetConfig,
    real_df: pd.DataFrame,
    real_profile: RealProfile,
    generated_info: GeneratedDatasetInfo,
    real_row_count: int,
    real_column_count: int,
    real_missing: Dict[str, Any],
    real_duplicates: Dict[str, Any],
) -> Dict[str, Any]:
    generated_df = load_dataframe(generated_info.path)

    schema_result = compute_schema_checks(real_profile, generated_df)

    missing_generated = compute_missingness(generated_df)
    duplicates_generated = compute_duplicates(generated_df, config.primary_key)

    numeric_cols = list(real_profile.numeric_ranges.keys())
    categorical_cols = list(real_profile.categorical_allowed.keys())
    text_cols = [
        col
        for col, t in real_profile.simplified_dtypes.items()
        if t == "object" and col not in numeric_cols and col not in categorical_cols
    ]

    numeric_validity = compute_numeric_validity(config, real_profile, generated_df)
    categorical_sanity = compute_categorical_sanity(real_profile, generated_df)
    outliers = compute_outliers(generated_df, numeric_cols)
    date_validation = compute_date_validation(config, real_profile, generated_df)
    row_constraints = evaluate_row_constraints(config, generated_df)

    # distribution similarity vs real data
    numeric_dist_sim = compute_numeric_distribution_similarity(real_df, generated_df, numeric_cols)
    categorical_dist_sim = compute_categorical_distribution_similarity(real_df, generated_df, categorical_cols)
    text_quality = compute_text_quality(real_df, generated_df, text_cols)

    num_ks = _safe_mean([v["ks_distance"] for v in numeric_dist_sim.values()])
    num_mean_diff = _safe_mean([v["mean_difference"] for v in numeric_dist_sim.values()])
    num_std_diff = _safe_mean([v["std_difference"] for v in numeric_dist_sim.values()])

    cat_js = _safe_mean([v["js_divergence"] for v in categorical_dist_sim.values()])

    txt_avg_len_real = _safe_mean([v["avg_length_real"] for v in text_quality.values()])
    txt_avg_len_gen = _safe_mean([v["avg_length_generated"] for v in text_quality.values()])
    txt_len_diff = None
    if txt_avg_len_real is not None and txt_avg_len_gen is not None:
        txt_len_diff = abs(txt_avg_len_real - txt_avg_len_gen)
    txt_empty_pct_gen = _safe_mean([v["empty_pct_generated"] for v in text_quality.values()])

    outlier_pct_mean = _safe_mean([v.get("count_pct") for v in outliers.values()])

    cat_unexpected_pct_mean = _safe_mean(
        [v.get("unexpected_value_pct") for v in categorical_sanity.values()]
    )

    date_invalid_pct_mean = _safe_mean(
        [v.get("invalid_values_pct") for v in date_validation.values()]
    )

    row_constraint_pcts = [v.get("violations_pct") for v in row_constraints.values()]
    row_constraint_mean_pct = _safe_mean(row_constraint_pcts)
    row_constraint_max_pct = (
        max([p for p in row_constraint_pcts if p is not None]) if row_constraint_pcts else None
    )

    summary = {
        "domain": generated_info.domain,
        "model": generated_info.model,
        "shot": generated_info.shot,
        "generated_path": str(generated_info.path.relative_to(PROJECT_ROOT)),
        "run_id": generated_info.run_id,
        "run_name": generated_info.run_name,
        "generated_row_count": len(generated_df),
        "generated_column_count": generated_df.shape[1],
        "generated_overall_missing_pct": missing_generated["overall_missing_pct"],
        "generated_duplicate_rows": duplicates_generated["duplicate_rows"],
        "generated_duplicate_rows_pct": duplicates_generated["duplicate_rows_pct"],
        "generated_duplicate_rows_primary_key": duplicates_generated.get("duplicate_rows_primary_key"),
        "generated_duplicate_rows_primary_key_pct": duplicates_generated.get("duplicate_rows_primary_key_pct"),
        "generated_type_correctness_pct": schema_result["type_correctness_pct"],
        "real_row_count": real_row_count,
        "real_column_count": real_column_count,
        "real_overall_missing_pct": real_missing["overall_missing_pct"],
        "real_duplicate_rows": real_duplicates["duplicate_rows"],
        "real_duplicate_rows_pct": real_duplicates["duplicate_rows_pct"],
        "real_duplicate_rows_primary_key": real_duplicates.get("duplicate_rows_primary_key"),
        "real_duplicate_rows_primary_key_pct": real_duplicates.get("duplicate_rows_primary_key_pct"),
        "numeric_avg_ks_real_vs_generated": num_ks,
        "numeric_avg_mean_diff_real_vs_generated": num_mean_diff,
        "numeric_avg_std_diff_real_vs_generated": num_std_diff,
        "categorical_avg_js_divergence_real_vs_generated": cat_js,
        "text_avg_length_real_mean": txt_avg_len_real,
        "text_avg_length_generated_mean": txt_avg_len_gen,
        "text_avg_length_abs_diff_mean": txt_len_diff,
        "text_empty_pct_generated_mean": txt_empty_pct_gen,
        "numeric_outlier_pct_mean": outlier_pct_mean,
        "categorical_unexpected_value_pct_mean": cat_unexpected_pct_mean,
        "date_invalid_values_pct_mean": date_invalid_pct_mean,
        "row_constraints_mean_violations_pct": row_constraint_mean_pct,
        "row_constraints_max_violations_pct": row_constraint_max_pct,
    }

    return summary


def run_quality_checks(
    domains: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    shots: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    domain_filter = {d.lower() for d in domains} if domains else None
    model_filter = {m.lower() for m in models} if models else None
    shot_filter = {s.lower() for s in shots} if shots else None

    summaries: List[Dict[str, Any]] = []
    real_metrics_records: List[Dict[str, Any]] = []

    for domain, config in DATASET_CONFIGS.items():
        if domain_filter and domain not in domain_filter:
            continue
        if not config.real_path.exists():
            print(f"Real dataset missing for domain '{domain}': {config.real_path}")
            continue
        try:
            real_df = load_dataframe(config.real_path)
        except Exception as exc:
            print(f"Failed to load real dataset '{config.real_path}': {exc}")
            continue

        real_row_count = len(real_df)
        real_column_count = real_df.shape[1]
        real_missing = compute_missingness(real_df)
        real_duplicates = compute_duplicates(real_df, config.primary_key)

        real_metrics_records.append(
            {
                "domain": domain,
                "row_count": real_row_count,
                "column_count": real_column_count,
                "overall_missing_pct": real_missing["overall_missing_pct"],
                "duplicate_rows": real_duplicates["duplicate_rows"],
                "duplicate_rows_pct": real_duplicates["duplicate_rows_pct"],
                "duplicate_rows_primary_key": real_duplicates.get("duplicate_rows_primary_key"),
                "duplicate_rows_primary_key_pct": real_duplicates.get("duplicate_rows_primary_key_pct"),
            }
        )

        real_profile = build_real_profile(config, real_df)
        generated_datasets = list_generated_datasets(config)
        if model_filter:
            generated_datasets = [
                ds for ds in generated_datasets if ds.model.lower() in model_filter
            ]
        if shot_filter:
            generated_datasets = [
                ds for ds in generated_datasets if ds.shot.lower() in shot_filter
            ]

        if not generated_datasets:
            print(f"No generated datasets found for domain '{domain}'.")
            continue

        for idx, generated_info in enumerate(generated_datasets, start=1):
            if limit is not None and len(summaries) >= limit:
                break

            print(
                f"Processing {domain} | model={generated_info.model} | shot={generated_info.shot} "
                f"({idx}/{len(generated_datasets)})"
            )
            try:
                summary = quality_checks_for_dataset(
                    config=config,
                    real_df=real_df,
                    real_profile=real_profile,
                    generated_info=generated_info,
                    real_row_count=real_row_count,
                    real_column_count=real_column_count,
                    real_missing=real_missing,
                    real_duplicates=real_duplicates,
                )
            except Exception as exc:
                print(f"Quality checks failed for {generated_info.path}: {exc}")
                continue

            summaries.append(summary)
            if limit is not None and len(summaries) >= limit:
                break

        if limit is not None and len(summaries) >= limit:
            break

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = QUALITY_OUTPUT_DIR / "summary_by_run.csv"
        summary_df.to_csv(summary_path, index=False)

        def aggregate_by(group_fields: Sequence[str], filename: str) -> None:
            if not all(field in summary_df.columns for field in group_fields):
                return
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                numeric_cols = [
                    col
                    for col in summary_df.columns
                    if col not in {"domain", "model", "shot", "run_id", "generated_path"}
                ]
            aggregated_rows: List[Dict[str, Any]] = []
            for keys, group in summary_df.groupby(list(group_fields)):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row: Dict[str, Any] = {}
                for field, value in zip(group_fields, keys):
                    row[field] = value
                row["runs_count"] = int(len(group))
                for col in numeric_cols:
                    if col in group_fields:
                        continue
                    values = pd.to_numeric(group[col], errors="coerce").dropna()
                    if values.empty:
                        continue
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std(ddof=0))
                    row[f"{col}_min"] = float(values.min())
                    row[f"{col}_max"] = float(values.max())
                aggregated_rows.append(row)
            if aggregated_rows:
                df = pd.DataFrame(aggregated_rows)
                path = QUALITY_OUTPUT_DIR / filename
                df.to_csv(path, index=False)

        aggregate_by(["domain"], "summary_by_domain.csv")
        aggregate_by(["model"], "summary_by_model.csv")
        aggregate_by(["shot"], "summary_by_shot.csv")
        aggregate_by(["domain", "model"], "summary_by_domain_model.csv")
        aggregate_by(["domain", "shot"], "summary_by_domain_shot.csv")
        aggregate_by(["model", "shot"], "summary_by_model_shot.csv")
        aggregate_by(["domain", "model", "run_name"], "summary_by_domain_model_run.csv")
        aggregate_by(["domain", "model", "shot"], "summary_by_domain_model_shot.csv")

    if real_metrics_records:
        real_df_metrics = (
            pd.DataFrame(real_metrics_records)
            .drop_duplicates(subset=["domain"], keep="last")
        )
        real_path = QUALITY_OUTPUT_DIR / "real_dataset_metrics.csv"
        real_df_metrics.to_csv(real_path, index=False)

    return summaries


def main(argv: Optional[Sequence[str]] = None) -> None:
    summaries = run_quality_checks()
    print(f"Completed quality checks for {len(summaries)} dataset(s).")
    if summaries:
        print("Latest report:")
        latest = summaries[-1]
        print(
            json.dumps(
                {
                    "domain": latest["domain"],
                    "model": latest["model"],
                    "shot": latest["shot"],
                    "run_name": latest.get("run_name"),
                    "generated_duplicate_rows_pct": latest.get("generated_duplicate_rows_pct"),
                    "real_duplicate_rows_pct": latest.get("real_duplicate_rows_pct"),
                    "generated_row_count": latest.get("generated_row_count"),
                    "numeric_avg_ks_real_vs_generated": latest.get("numeric_avg_ks_real_vs_generated"),
                    "categorical_avg_js_divergence_real_vs_generated": latest.get("categorical_avg_js_divergence_real_vs_generated"),
                },
                indent=2,
            )
        )

def inspect_type_correctness(model: str, domain: str, shot: str, run_id: str | None = None, show_details: bool = True):
    """Inspect column-level type correctness for a single model/domain/shot/run."""
    from dataset_quality_checks import (
        DATASET_CONFIGS,
        load_dataframe,
        build_real_profile,
        compute_schema_checks,
        list_generated_datasets,
        simplify_dtype,
    )

    config = DATASET_CONFIGS.get(domain)
    if not config:
        print(f"Unknown domain: {domain}")
        return None

    if not config.real_path.exists():
        print(f"Real dataset not found: {config.real_path}")
        return None

    real_df = load_dataframe(config.real_path)
    real_profile = build_real_profile(config, real_df)

    generated_datasets = list_generated_datasets(config)
    matching = [
        ds for ds in generated_datasets
        if ds.model.lower() == model.lower() and ds.shot.lower() == shot.lower()
    ]
    if not matching:
        print(f"No generated dataset found for model={model}, domain={domain}, shot={shot}")
        return None

    if run_id:
        matching = [ds for ds in matching if ds.run_id == run_id]
        if not matching:
            print(f"No dataset found for run_id={run_id}")
            return None

    generated_info = matching[0]
    if len(matching) > 1:
        print(f"Multiple runs found, using: {generated_info.run_name}")
        print(f"Available runs: {[ds.run_id for ds in matching]}")

    try:
        generated_df = load_dataframe(generated_info.path)
    except Exception as exc:
        print(f"Failed to load generated dataset: {exc}")
        return None

    schema_result = compute_schema_checks(real_profile, generated_df)

    if show_details:
        print("=" * 80)
        print("DATA TYPE CORRECTNESS INSPECTION")
        print("=" * 80)
        print(f"Model: {model}")
        print(f"Domain: {domain}")
        print(f"Shot: {shot}")
        print(f"Run: {generated_info.run_id} ({generated_info.run_name})")
        print(f"Generated Dataset: {generated_info.path.relative_to(PROJECT_ROOT)}")
        print(f"\nType Correctness: {schema_result['type_correctness_pct']:.2f}%")
        print(f"Expected Columns: {schema_result['expected_column_count']}")
        print(f"Matched Types: {schema_result['expected_column_count'] - len(schema_result['mismatched_columns'])}")
        print(f"Mismatched Types: {len(schema_result['mismatched_columns'])}")
        print(f"Extra Columns: {len(schema_result['extra_columns'])}")
        print("\n" + "=" * 80)
        print("TYPE COMPARISON (Expected vs Observed)")
        print("=" * 80)

    generated_types = {col: simplify_dtype(dtype) for col, dtype in generated_df.dtypes.items()}
    actual_dtypes = {col: str(dtype) for col, dtype in generated_df.dtypes.items()}

    comparison_rows = []
    for col in sorted(real_profile.simplified_dtypes.keys()):
        expected = real_profile.simplified_dtypes[col]
        observed = generated_types.get(col, "MISSING")
        actual_dtype = actual_dtypes.get(col, "N/A")
        status = "OK" if expected == observed else "MISMATCH"
        comparison_rows.append({
            "Column": col,
            "Expected": expected,
            "Observed": observed,
            "Actual Pandas Dtype": actual_dtype,
            "Status": status,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    if show_details:
        print(comparison_df.to_string(index=False))

    if schema_result['mismatched_columns'] and show_details:
        print("\n" + "=" * 80)
        print("TYPE MISMATCHES")
        print("=" * 80)
        for mismatch in schema_result['mismatched_columns']:
            col = mismatch['column']
            issue = mismatch['issue']
            expected = mismatch['expected']
            observed = mismatch.get('observed', 'MISSING')
            actual_dtype = actual_dtypes.get(col, 'N/A')
            print(f"\nColumn: {col}")
            print(f"  Issue: {issue}")
            print(f"  Expected Type: {expected}")
            print(f"  Observed Type: {observed}")
            print(f"  Actual Pandas Dtype: {actual_dtype}")
            if col in generated_df.columns:
                sample_vals = generated_df[col].dropna().head(3).tolist()
                print(f"  Sample Values: {sample_vals}")

    if schema_result['extra_columns'] and show_details:
        print("\n" + "=" * 80)
        print("EXTRA COLUMNS (in generated but not in expected schema)")
        print("=" * 80)
        for col in schema_result['extra_columns']:
            dtype = generated_types.get(col, 'unknown')
            actual_dtype = actual_dtypes.get(col, 'N/A')
            print(f"  {col}: {dtype} (pandas: {actual_dtype})")

    return {
        "schema_result": schema_result,
        "comparison_df": comparison_df,
        "real_profile": real_profile,
        "generated_df": generated_df,
        "generated_types": generated_types,
    }
    

def inspect_type_correctness_runs(
    model: str,
    domain: str,
    shot: str,
    run_ids: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    from dataset_quality_checks import DATASET_CONFIGS, list_generated_datasets

    config = DATASET_CONFIGS.get(domain)
    if not config:
        print(f"Unknown domain: {domain}")
        return pd.DataFrame()

    generated_datasets = list_generated_datasets(config)
    matching = [
        ds for ds in generated_datasets
        if ds.model.lower() == model.lower() and ds.shot.lower() == shot.lower()
    ]

    if not matching:
        print(f"No generated datasets found for {model} | {domain} | {shot}")
        return pd.DataFrame()

    run_map = {}
    for ds in matching:
        run_map.setdefault(ds.run_id, ds)

    if run_ids:
        run_map = {run_id: run_map[run_id] for run_id in run_ids if run_id in run_map}
        missing = [run_id for run_id in run_ids if run_id not in run_map]
        if missing:
            print(f"Runs not found for {model} | {domain} | {shot}: {missing}")

    if not run_map:
        print(f"No matching runs left after filtering for {model} | {domain} | {shot}")
        return pd.DataFrame()

    rows = []
    column_mismatches_by_run = {}
    run_comparisons: dict[str, pd.DataFrame] = {}
    
    for run_id in sorted(run_map.keys()):
        ds = run_map[run_id]
        result = inspect_type_correctness(
            model=model,
            domain=domain,
            shot=shot,
            run_id=run_id,
            show_details=verbose,
        )
        if not result:
            continue
        
        rows.append({
            "run_id": run_id,
            "run_name": ds.run_name,
            "generated_type_correctness_pct": result['schema_result']['type_correctness_pct'],
        })
        
        comparison_df = result['comparison_df'].copy()
        run_comparisons[run_id] = comparison_df
        
        for _, row in comparison_df.iterrows():
            col = row['Column']
            if col not in column_mismatches_by_run:
                column_mismatches_by_run[col] = {}
            column_mismatches_by_run[col][run_id] = {
                'expected': row['Expected'],
                'observed': row['Observed'],
                'status': row['Status'],
            }

    if not rows:
        print(f"Could not compute type correctness for any runs of {model} | {domain} | {shot}")
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    summary_df['generated_type_correctness_pct'] = summary_df['generated_type_correctness_pct'].round(4)

    avg_pct = summary_df['generated_type_correctness_pct'].mean()
    print("\n" + "=" * 80)
    print(f"Type correctness by run for {model} | {domain} | {shot}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nAverage correctness across {len(summary_df)} run(s): {avg_pct:.2f}%")
    
    if run_comparisons:
        print("\n" + "=" * 80)
        print("Per-run column types vs real dataset")
        print("=" * 80)
        for run_id in sorted(run_comparisons.keys()):
            run_name = summary_df.loc[summary_df['run_id'] == run_id, 'run_name'].iloc[0]
            table = run_comparisons[run_id][['Column', 'Expected', 'Observed', 'Status']].rename(
                columns={
                    'Expected': 'Real type',
                    'Observed': 'Observed type'
                }
            )
            print(f"\nRun {run_id} ({run_name})")
            print(table.to_string(index=False))
    
    return summary_df



if __name__ == "__main__":
    main()