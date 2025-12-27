'''LLM Dataset Generator using HuggingFace InferenceClient'''

### IMPORTS ###

import os
import io
import re
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################


# import the modern HuggingFace client
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except ImportError:
    HF_CLIENT_AVAILABLE = False
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    
  
# configuration parameters
class Config:
    CHUNK_SIZE = 50
    TARGET_ROWS = 1000
    MAX_RETRIES = 3
    MAX_FAILED_STREAK = 3
    RETRY_DELAY = 2
    BASE_OUTPUT_DIR = Path("../../data/generated")


#schemas for the datasets with their columns, years and descriptions
SCHEMAS: Dict[str, Dict[str, str]] = {
    "hatecrime": {
        "columns": "data_year,agency_type_name,state_abbr,state_name,incident_date,adult_victim_count,juvenile_victim_count,total_offender_count,adult_offender_count,juvenile_offender_count,offender_race,offender_ethnicity,victim_count,offense_name,total_individual_victims,location_name,bias_desc,victim_types,multiple_offense,multiple_bias",
        "years": "1991-2014",
        "description": "FBI Hate Crime Statistics"
    },
    "employment": {
        "columns": "EmployerName,SicCodes,DiffMeanHourlyPercent,DiffMedianHourlyPercent,DiffMeanBonusPercent,DiffMedianBonusPercent,MaleBonusPercent,FemaleBonusPercent,MaleLowerQuartile,FemaleLowerQuartile,MaleLowerMiddleQuartile,FemaleLowerMiddleQuartile,MaleUpperMiddleQuartile,FemaleUpperMiddleQuartile,MaleTopQuartile,FemaleTopQuartile,EmployerSize,DateSubmitted",
        "years": "2024-2025",
        "description": "UK Gender Pay Gap Data"
    },
    "lending": {
        "columns": "activity_year,derived_msa-md,state_code,county_code,census_tract,derived_loan_product_type,derived_dwelling_category,derived_ethnicity,derived_race,derived_sex,action_taken,loan_purpose,loan_amount,loan_to_value_ratio,interest_rate,rate_spread,hoepa_status,total_loan_costs,total_points_and_fees,origination_charges,discount_points,lender_credits,loan_term,prepayment_penalty_term,intro_rate_period,negative_amortization,interest_only_payment,balloon_payment,other_nonamortizing_features,property_value,construction_method,occupancy_type,manufactured_home_secured_property_type,manufactured_home_land_property_interest,total_units,multifamily_affordable_units,income,debt_to_income_ratio,applicant_credit_score_type,co-applicant_credit_score_type,applicant_ethnicity-1,co-applicant_ethnicity-1,applicant_ethnicity_observed,co-applicant_ethnicity_observed,applicant_race-1,co-applicant_race-1,applicant_race_observed,co-applicant_race_observed,applicant_sex,co-applicant_sex,applicant_sex_observed,co-applicant_sex_observed,applicant_age,co-applicant_age,applicant_age_above_62,co-applicant_age_above_62,submission_of_application,initially_payable_to_institution",
        "years": "2024",
        "description": "HMDA Mortgage Lending Data"
    }
}


def clean_llm_csv_output(text: str) -> str:
    '''Extract raw CSV-like content without altering commas inside quotes.'''
    
    fence = re.search(r"```(?:csv|text)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1)
    else:
        text = re.sub(r"```(?:csv|json|text)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)


    text = text.replace("\uFEFF", "").strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # keeping only lines that look like CSV, containing at least one comma
    lines = [ln.rstrip() for ln in text.splitlines()]
    csv_lines = [ln for ln in lines if ln and "," in ln]
    
    return "\n".join(csv_lines)


def parse_llm_csv(text: str, domain: Optional[str] = None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    '''Minimal parsing that preserves all rows and tracks schema corrections.'''
    
    cleaned_text = clean_llm_csv_output(text)
    if not cleaned_text:
        return pd.DataFrame(), {"empty": True}

    reader = csv.reader(io.StringIO(cleaned_text))
    rows = [[cell.strip() for cell in row] for row in reader]
    rows = [r for r in rows if any(c != "" for c in r)]
    if not rows:
        return pd.DataFrame(), {"empty": True}

    expected_cols: Optional[List[str]] = None
    if domain and domain in SCHEMAS:
        expected_cols = SCHEMAS[domain]["columns"].split(",")

    from collections import Counter
    lengths = [len(r) for r in rows if len(r) > 0]
    target_len = len(expected_cols) if expected_cols else Counter(lengths).most_common(1)[0][0]

    def normalize_token(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    def looks_like_header(cells: List[str]) -> bool:
        # prefering name-based match when we know the expected schema
        if expected_cols and len(cells) == len(expected_cols):
            exp_norm = set(normalize_token(c) for c in expected_cols)
            cells_norm = [normalize_token(c) for c in cells]
            matches = sum(1 for c in cells_norm if c in exp_norm)
            # if names match, then it's a header
            if matches >= max(4, int(0.6 * len(expected_cols))):
                return True
            alpha_like = sum(1 for c in cells if re.search(r"[A-Za-z]", c) and not re.match(r"^\s*[-+]?\d", c))
            if (alpha_like / max(1, len(cells))) >= 0.7:
                return True
            return False
        
        alpha_like = sum(1 for c in cells if re.search(r"[A-Za-z]", c) and not re.match(r"^\s*[-+]?\d", c))
        return (alpha_like / max(1, len(cells))) >= 0.8

    first_idx = next((i for i, r in enumerate(rows) if len(r) > 0), 0)
    header_present = looks_like_header(rows[first_idx])

    data_rows = rows[first_idx + 1:] if header_present else rows[first_idx:]

    # mapping columns to expected schema positions
    normalized_rows: List[List[str]] = []
    for r in data_rows:
        if expected_cols and len(expected_cols) == target_len:
            # mapping to expected schema
            mapped_row = [""] * target_len
            
            if header_present and len(r) == len(rows[first_idx]):
                original_headers = rows[first_idx]
                name_matches = 0
                for i, orig_col in enumerate(original_headers):
                    if i < len(r):
                        # finding matching expected column
                        matched = False
                        for j, exp_col in enumerate(expected_cols):
                            if normalize_token(orig_col) == normalize_token(exp_col):
                                mapped_row[j] = r[i]
                                name_matches += 1
                                matched = True
                                break
                            
                if name_matches < max(4, int(0.4 * len(expected_cols))):
                    for i, value in enumerate(r):
                        if i < target_len:
                            mapped_row[i] = value
            else:
                for i, value in enumerate(r):
                    if i < target_len:
                        mapped_row[i] = value
                    else:
                        break 
            
            normalized_rows.append(mapped_row)
        else:
            if len(r) == target_len:
                normalized_rows.append(r)
            elif len(r) > target_len:
                normalized_rows.append(r[:target_len])
            else:
                normalized_rows.append(r + [""] * (target_len - len(r)))

    if not normalized_rows:
        return pd.DataFrame(), {"empty": True}

    # tracking what we are doing with columns
    parse_info = {
        "header_present": header_present,
        "original_columns": rows[first_idx] if header_present else None,
        "expected_columns": expected_cols,
        "used_expected_schema": bool(expected_cols and len(expected_cols) == target_len),
        "schema_correction_needed": False,
        "correction_type": None,
        "corrections_made": []
    }

    # checking if we need to correct the schema
    if expected_cols and len(expected_cols) == target_len:
        if header_present:
            original_cols = rows[first_idx]
            if original_cols != expected_cols:
                parse_info["schema_correction_needed"] = True
                parse_info["correction_type"] = "column_rename"
                parse_info["corrections_made"].append(f"Renamed columns to match expected schema")
        else:
            # no header but we are using expected schema 
            parse_info["schema_correction_needed"] = True
            parse_info["correction_type"] = "add_header_and_columns"
            parse_info["corrections_made"].append(f"Added expected schema columns (no header in LLM output)")

    columns = expected_cols if (expected_cols and len(expected_cols) == target_len) else [f"col_{i}" for i in range(target_len)]
    df = pd.DataFrame(normalized_rows, columns=columns)
    df = df.dropna(how="all")
    
    return df, parse_info


def clean_llm_json_output(text: str) -> str:
    '''Extract a the best effort JSON array from messy LLM output.'''
    if not isinstance(text, str):
        return ""

    # removing common code fences first
    text = re.sub(r"```(?:json|text)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.replace("\uFEFF", "").strip()

    match = re.search(r"\{\s*\"data\"\s*:\s*(\[[\s\S]*?\])\s*\}", text)
    if match:
        text = match.group(1)

    if "[" in text and "]" in text and text.find("[") < text.rfind("]"):
        start, end = text.find("["), text.rfind("]")
        candidate = text[start:end+1]
        candidate = re.sub(r",\s*\]", "]", candidate) 
        # prefering double quotes, if single quotes dominate, convert singles to doubles
        if candidate.count('"') < candidate.count("'"):
            candidate = re.sub(r"(?<!\\)'", '"', candidate)
        return candidate.strip()

    # trying to build an array from lines with JSON looking objects
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    text_fixed = re.sub(r"}\s*{", "},\n{", text)
    lines_fixed = [ln.strip() for ln in text_fixed.splitlines() if ln.strip()]

    def try_parse_array(raw: str) -> Optional[str]:
        raw = raw.strip()
        if raw.count('"') < raw.count("'"):
            raw = re.sub(r"(?<!\\)'", '"', raw)
        raw = re.sub(r",\s*(\]|\})", r"\1", raw)
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return json.dumps(obj)
            if isinstance(obj, dict):
                return json.dumps([obj])
        except Exception:
            pass
        return None

    attempt = try_parse_array(text_fixed)
    if attempt:
        return attempt

    objs = []
    for ln in lines_fixed:
        if not (ln.startswith("{") and ln.endswith("}")):
            continue
        ln_norm = re.sub(r"(?<!\\)'", '"', ln)
        ln_norm = re.sub(r",\s*\}$", "}", ln_norm)
        try:
            obj = json.loads(ln_norm)
            if isinstance(obj, dict):
                objs.append(obj)
        except Exception:
            try:
                maybe = ast.literal_eval(ln)
                if isinstance(maybe, dict):
                    objs.append(maybe)
            except Exception:
                continue
    if objs:
        return json.dumps(objs)

    return ""


def parse_llm_json(text: str) -> pd.DataFrame:
    '''Parse json output from the LLM'''
    
    cleaned = clean_llm_json_output(text)
    if not cleaned:
        return pd.DataFrame()

    try:
        data = json.loads(cleaned)
    except Exception:
        return pd.DataFrame()

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return pd.DataFrame()

    try:
        df = pd.json_normalize(data, sep=".")
    except Exception:
        try:
            df = pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    df = df.dropna(how="all").drop_duplicates()
    return df


def validate_and_fix_schema(df: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, Dict]:
    '''Fix column schema mismatches'''
    
    schema_issues = {
        "had_schema_mismatch": False,
        "original_columns": list(df.columns) if not df.empty else [],
        "expected_columns": SCHEMAS[domain]["columns"].split(","),
        "correction_type": None,
        "corrections_made": []
    }
    
    if df.empty:
        return df, schema_issues
    
    expected_cols = SCHEMAS[domain]["columns"].split(",")
    actual_cols = list(df.columns)
    
    # checking if schema is already perfect
    if actual_cols == expected_cols:
        return df, schema_issues
    
    # schema mismatch detected
    schema_issues["had_schema_mismatch"] = True
    
    # case 1: same number of columns but different names
    if len(actual_cols) == len(expected_cols):
        schema_issues["correction_type"] = "column_rename"
        schema_issues["corrections_made"].append(f"Renamed {len(actual_cols)} columns to match expected schema")
        df.columns = expected_cols
        return df, schema_issues
    
    # case 2: missing some columns
    elif len(actual_cols) < len(expected_cols):
        schema_issues["correction_type"] = "add_missing_columns"
        missing_cols = set(expected_cols) - set(actual_cols)
        
        for col in missing_cols:
            df[col] = ""  # adding empty columns for missing ones
            schema_issues["corrections_made"].append(f"Added missing column: {col}")
        
        df = df[expected_cols]
        return df, schema_issues
    
    # case 3: too many columns
    else:
        schema_issues["correction_type"] = "remove_extra_columns"
        
        matched_cols = [col for col in expected_cols if col in actual_cols]
        
        if len(matched_cols) >= len(expected_cols) * 0.7:  # If we can match 70%+ by name
            df_matched = df[matched_cols].copy()
            # adding missing columns
            for col in expected_cols:
                if col not in matched_cols:
                    df_matched[col] = ""
                    schema_issues["corrections_made"].append(f"Added missing column: {col}")
            df = df_matched[expected_cols]
            schema_issues["corrections_made"].append(f"Kept {len(matched_cols)} matching columns, removed {len(actual_cols) - len(matched_cols)} extra")
        else:
            df_subset = df.iloc[:, :len(expected_cols)].copy()
            df_subset.columns = expected_cols
            schema_issues["corrections_made"].append(f"Used first {len(expected_cols)} columns positionally, removed {len(actual_cols) - len(expected_cols)} extra")
            df = df_subset
        
        return df, schema_issues
    
    return df, schema_issues


######### EXAMPLE DATA #########

def get_examples(domain: str) -> List[Dict]:
    '''Get example rows for prompting.'''
    
    examples = {
        "hatecrime": [
            {
                "data_year": 2004,
                "agency_type_name": "City",
                "state_abbr": "NJ",
                "state_name": "New Jersey",
                "incident_date": "2004-05-13",
                "adult_victim_count": null,
                "juvenile_victim_count": null,
                "total_offender_count": 0,
                "adult_offender_count": null,
                "juvenile_offender_count": null,
                "offender_race": "Unknown",
                "offender_ethnicity": "Not Specified",
                "victim_count": 1,
                "offense_name": "Destruction/Damage/Vandalism of Property",
                "total_individual_victims": 0.0,
                "location_name": "School/College",
                "bias_desc": "Anti-Black or African American",
                "victim_types": "Other",
                "multiple_offense": "S",
                "multiple_bias": "S"
            },
            {
                "data_year": 2022,
                "agency_type_name": "Other",
                "state_abbr": "MN",
                "state_name": "Minnesota",
                "incident_date": "2022-07-24",
                "adult_victim_count": 2.0,
                "juvenile_victim_count": 0.0,
                "total_offender_count": 1,
                "adult_offender_count": 1.0,
                "juvenile_offender_count": 0.0,
                "offender_race": "Black or African American",
                "offender_ethnicity": "Not Hispanic or Latino",
                "victim_count": 1,
                "offense_name": "Simple Assault",
                "total_individual_victims": 2.0,
                "location_name": "Air/Bus/Train Terminal",
                "bias_desc": "Anti-Gay (Male)",
                "victim_types": "Individual",
                "multiple_offense": "S",
                "multiple_bias": "S"
            },
            {
                "data_year": 1997,
                "agency_type_name": "City",
                "state_abbr": "CA",
                "state_name": "California",
                "incident_date": "1997-05-04",
                "adult_victim_count": null,
                "juvenile_victim_count": null,
                "total_offender_count": 1,
                "adult_offender_count": null,
                "juvenile_offender_count": null,
                "offender_race": "White",
                "offender_ethnicity": "Not Specified",
                "victim_count": 1,
                "offense_name": "Aggravated Assault",
                "total_individual_victims": 1.0,
                "location_name": "Highway/Road/Alley/Street/Sidewalk",
                "bias_desc": "Anti-Asian",
                "victim_types": "Individual",
                "multiple_offense": "S",
                "multiple_bias": "S"
            }

        ],
        "employment": [
            {
                "EmployerName": "MISHCON DE REYA GROUP (OPERATIONS) LIMITED",
                "SicCodes": "64209",
                "DiffMeanHourlyPercent": 30.4,
                "DiffMedianHourlyPercent": 23.6,
                "DiffMeanBonusPercent": 73.9,
                "DiffMedianBonusPercent": 30.0,
                "MaleBonusPercent": 55.4,
                "FemaleBonusPercent": 64.0,
                "MaleLowerQuartile": 25.4,
                "FemaleLowerQuartile": 74.6,
                "MaleLowerMiddleQuartile": 15.1,
                "FemaleLowerMiddleQuartile": 84.9,
                "MaleUpperMiddleQuartile": 27.0,
                "FemaleUpperMiddleQuartile": 73.0,
                "MaleTopQuartile": 55.6,
                "FemaleTopQuartile": 44.4,
                "EmployerSize": "500-999",
                "DateSubmitted": "2025-04-03 15:10:41"
            },
            {
                "EmployerName": "DE VERE WOKEFIELD ESTATE LIMITED",
                "SicCodes": "55100",
                "DiffMeanHourlyPercent": 16.7,
                "DiffMedianHourlyPercent": 3.6,
                "DiffMeanBonusPercent": 40.5,
                "DiffMedianBonusPercent": -119.2,
                "MaleBonusPercent": 42.3,
                "FemaleBonusPercent": 49.7,
                "MaleLowerQuartile": 29.9,
                "FemaleLowerQuartile": 70.1,
                "MaleLowerMiddleQuartile": 45.6,
                "FemaleLowerMiddleQuartile": 54.4,
                "MaleUpperMiddleQuartile": 44.8,
                "FemaleUpperMiddleQuartile": 55.2,
                "MaleTopQuartile": 63.2,
                "FemaleTopQuartile": 36.8,
                "EmployerSize": "250-499",
                "DateSubmitted": "2025-03-31 14:44:02"
            },
            {
                "EmployerName": "ODEON CINEMAS LIMITED",
                "SicCodes": "59131,\n59140",
                "DiffMeanHourlyPercent": 9.4,
                "DiffMedianHourlyPercent": 0.0,
                "DiffMeanBonusPercent": 37.3,
                "DiffMedianBonusPercent": 9.6,
                "MaleBonusPercent": 18.8,
                "FemaleBonusPercent": 17.0,
                "MaleLowerQuartile": 54.0,
                "FemaleLowerQuartile": 46.0,
                "MaleLowerMiddleQuartile": 54.0,
                "FemaleLowerMiddleQuartile": 46.0,
                "MaleUpperMiddleQuartile": 52.0,
                "FemaleUpperMiddleQuartile": 48.0,
                "MaleTopQuartile": 57.0,
                "FemaleTopQuartile": 43.0,
                "EmployerSize": "1000-4999",
                "DateSubmitted": "2025-04-04 11:41:38"
            }

        ],
        "lending": [
            {
                "activity_year": 2024,
                "derived_msa-md": 99999,
                "state_code": "CT",
                "county_code": 9150.0,
                "census_tract": 9150708100.0,
                "derived_loan_product_type": "Conventional:Subordinate Lien",
                "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
                "derived_ethnicity": "Not Hispanic or Latino",
                "derived_race": "White",
                "derived_sex": "Male",
                "action_taken": "File closed for incompleteness",
                "loan_purpose": 32,
                "loan_amount": 25000.0,
                "loan_to_value_ratio": null,
                "interest_rate": null,
                "rate_spread": null,
                "hoepa_status": 3,
                "total_loan_costs": null,
                "total_points_and_fees": null,
                "origination_charges": null,
                "discount_points": null,
                "lender_credits": null,
                "loan_term": "240",
                "prepayment_penalty_term": null,
                "intro_rate_period": null,
                "negative_amortization": 2,
                "interest_only_payment": 2,
                "balloon_payment": 2,
                "other_nonamortizing_features": 2,
                "property_value": null,
                "construction_method": 1,
                "occupancy_type": 1,
                "manufactured_home_secured_property_type": 3,
                "manufactured_home_land_property_interest": 5,
                "total_units": 1,
                "multifamily_affordable_units": null,
                "income": 96.0,
                "debt_to_income_ratio": null,
                "applicant_credit_score_type": 1111,
                "co-applicant_credit_score_type": 1111,
                "applicant_ethnicity-1": 2.0,
                "co-applicant_ethnicity-1": 5.0,
                "applicant_ethnicity_observed": 2,
                "co-applicant_ethnicity_observed": 4,
                "applicant_race-1": 5.0,
                "co-applicant_race-1": 8.0,
                "applicant_race_observed": 2,
                "co-applicant_race_observed": 4,
                "applicant_sex": 1,
                "co-applicant_sex": 5,
                "applicant_sex_observed": 2,
                "co-applicant_sex_observed": 4,
                "applicant_age": "65-74",
                "co-applicant_age": "9999",
                "applicant_age_above_62": "Yes",
                "co-applicant_age_above_62": null,
                "submission_of_application": 1,
                "initially_payable_to_institution": 1
            },
            {
                "activity_year": 2024,
                "derived_msa-md": 99999,
                "state_code": "PA",
                "county_code": 42127.0,
                "census_tract": 42127960600.0,
                "derived_loan_product_type": "VA:First Lien",
                "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
                "derived_ethnicity": "Ethnicity Not Available",
                "derived_race": "Race Not Available",
                "derived_sex": "Sex Not Available",
                "action_taken": "Application denied",
                "loan_purpose": 32,
                "loan_amount": 355000.0,
                "loan_to_value_ratio": null,
                "interest_rate": null,
                "rate_spread": null,
                "hoepa_status": 3,
                "total_loan_costs": null,
                "total_points_and_fees": null,
                "origination_charges": null,
                "discount_points": null,
                "lender_credits": null,
                "loan_term": "360",
                "prepayment_penalty_term": null,
                "intro_rate_period": null,
                "negative_amortization": 2,
                "interest_only_payment": 2,
                "balloon_payment": 2,
                "other_nonamortizing_features": 2,
                "property_value": null,
                "construction_method": 1,
                "occupancy_type": 1,
                "manufactured_home_secured_property_type": 3,
                "manufactured_home_land_property_interest": 5,
                "total_units": 1,
                "multifamily_affordable_units": null,
                "income": 108.0,
                "debt_to_income_ratio": null,
                "applicant_credit_score_type": 3,
                "co-applicant_credit_score_type": 10,
                "applicant_ethnicity-1": 3.0,
                "co-applicant_ethnicity-1": 5.0,
                "applicant_ethnicity_observed": 2,
                "co-applicant_ethnicity_observed": 4,
                "applicant_race-1": 6.0,
                "co-applicant_race-1": 8.0,
                "applicant_race_observed": 2,
                "co-applicant_race_observed": 4,
                "applicant_sex": 3,
                "co-applicant_sex": 5,
                "applicant_sex_observed": 2,
                "co-applicant_sex_observed": 4,
                "applicant_age": "65-74",
                "co-applicant_age": "9999",
                "applicant_age_above_62": "Yes",
                "co-applicant_age_above_62": null,
                "submission_of_application": 1,
                "initially_payable_to_institution": 1
            },
            {
                "activity_year": 2024,
                "derived_msa-md": 12054,
                "state_code": "GA",
                "county_code": 13121.0,
                "census_tract": 13121010529.0,
                "derived_loan_product_type": "FHA:First Lien",
                "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
                "derived_ethnicity": "Ethnicity Not Available",
                "derived_race": "Race Not Available",
                "derived_sex": "Sex Not Available",
                "action_taken": "6",
                "loan_purpose": 32,
                "loan_amount": 225000.0,
                "loan_to_value_ratio": null,
                "interest_rate": 6.875,
                "rate_spread": null,
                "hoepa_status": 2,
                "total_loan_costs": "16692.05",
                "total_points_and_fees": null,
                "origination_charges": "10197.8",
                "discount_points": "8302.8",
                "lender_credits": null,
                "loan_term": "300.0",
                "prepayment_penalty_term": null,
                "intro_rate_period": null,
                "negative_amortization": 2,
                "interest_only_payment": 2,
                "balloon_payment": 2,
                "other_nonamortizing_features": 2,
                "property_value": 275000.0,
                "construction_method": 1,
                "occupancy_type": 1,
                "manufactured_home_secured_property_type": 3,
                "manufactured_home_land_property_interest": 5,
                "total_units": 1,
                "multifamily_affordable_units": null,
                "income": null,
                "debt_to_income_ratio": null,
                "applicant_credit_score_type": 9,
                "co-applicant_credit_score_type": 9,
                "applicant_ethnicity-1": 4.0,
                "co-applicant_ethnicity-1": 4.0,
                "applicant_ethnicity_observed": 3,
                "co-applicant_ethnicity_observed": 3,
                "applicant_race-1": 7.0,
                "co-applicant_race-1": 7.0,
                "applicant_race_observed": 3,
                "co-applicant_race_observed": 3,
                "applicant_sex": 4,
                "co-applicant_sex": 4,
                "applicant_sex_observed": 3,
                "co-applicant_sex_observed": 3,
                "applicant_age": "8888",
                "co-applicant_age": "8888",
                "applicant_age_above_62": null,
                "co-applicant_age_above_62": null,
                "submission_of_application": 3,
                "initially_payable_to_institution": 3
            }
        ]
    }
    return examples.get(domain, [])


def get_csv_examples(domain: str) -> str:
    """Get CSV formatted example rows"""
    
    
    csv_examples = {
        "hatecrime": """2004,City,NJ,New Jersey,2004-05-13,,,0,,,Unknown,Not Specified,1,Destruction/Damage/Vandalism of Property,0.0,School/College,Anti-Black or African American,Other,S,S
2022,Other,MN,Minnesota,2022-07-24,2.0,0.0,1,1.0,0.0,Black or African American,Not Hispanic or Latino,1,Simple Assault,2.0,Air/Bus/Train Terminal,Anti-Gay (Male),Individual,S,S
1997,City,CA,California,1997-05-04,,,1,,,White,Not Specified,1,Aggravated Assault,1.0,Highway/Road/Alley/Street/Sidewalk,Anti-Asian,Individual,S,S""",
        
        "employment": """MISHCON DE REYA GROUP (OPERATIONS) LIMITED,64209,30.4,23.6,73.9,30.0,55.4,64.0,25.4,74.6,15.1,84.9,27.0,73.0,55.6,44.4,500-999,2025-04-03 15:10:41
DE VERE WOKEFIELD ESTATE LIMITED,55100,16.7,3.6,40.5,-119.2,42.3,49.7,29.9,70.1,45.6,54.4,44.8,55.2,63.2,36.8,250-499,2025-03-31 14:44:02
ODEON CINEMAS LIMITED,"59131,59140",9.4,0.0,37.3,9.6,18.8,17.0,54.0,46.0,54.0,46.0,52.0,48.0,57.0,43.0,1000-4999,2025-04-04 11:41:38""",
        
        "lending": """2024,99999,CT,9150.0,9150708100.0,Conventional:Subordinate Lien,Single Family (1-4 Units):Site-Built,Not Hispanic or Latino,White,Male,File closed for incompleteness,32,25000.0,,,,3,,,,,,240,,,2,2,2,2,,1,1,3,5,1,,96.0,,1111,1111,2.0,5.0,2,4,5.0,8.0,2,4,1,5,2,4,65-74,9999,Yes,,1,1
2024,99999,PA,42127.0,42127960600.0,VA:First Lien,Single Family (1-4 Units):Site-Built,Ethnicity Not Available,Race Not Available,Sex Not Available,Application denied,32,355000.0,,,,3,,,,,,360,,,2,2,2,2,,1,1,3,5,1,,108.0,,3,10,3.0,5.0,2,4,6.0,8.0,2,4,3,5,2,4,65-74,9999,Yes,,1,1
2024,12054,GA,13121.0,13121010529.0,FHA:First Lien,Single Family (1-4 Units):Site-Built,Ethnicity Not Available,Race Not Available,Sex Not Available,6,32,225000.0,,6.875,,2,16692.05,,10197.8,8302.8,,300.0,,,2,2,2,2,275000.0,1,1,3,5,1,,,,9,9,4.0,4.0,3,3,7.0,7.0,3,3,4,4,3,3,8888,8888,,,3,3"""
    }
    return csv_examples.get(domain, "")


######### PROMPT BUILDING #########


def build_prompt(domain: str, format_type: str, shot_type: str, chunk_num: int = 1) -> str:
    """Build generation prompt"""
    schema_info = SCHEMAS[domain]
    
    if format_type == "csv":
        prompt = f"""Generate {Config.CHUNK_SIZE} rows of {domain} data in CSV format.

DOMAIN: {schema_info['description']}
TIME PERIOD: {schema_info['years']}
REQUIRED COLUMNS: {schema_info['columns']}

CRITICAL: Output ONLY the CSV data with NO explanations, NO code blocks, NO commentary.
Start with the header row, then exactly {Config.CHUNK_SIZE} data rows.

"""
        
        # adding CSV examples based on shot type
        if shot_type == "one":
            csv_examples = get_csv_examples(domain)
            if csv_examples:
                first_line = csv_examples.split('\n')[0]
                prompt += f"EXAMPLE ROW:\n{first_line}\n\n"
        elif shot_type == "few":
            csv_examples = get_csv_examples(domain)
            if csv_examples:
                prompt += f"EXAMPLE ROWS:\n{csv_examples}\n\n"
    
    else:  # json
        prompt = f"""Generate {Config.CHUNK_SIZE} rows of {domain} data as JSON array.

DOMAIN: {schema_info['description']}
TIME PERIOD: {schema_info['years']}
REQUIRED COLUMNS: {schema_info['columns']}

CRITICAL: Output ONLY a JSON array with NO explanations, NO code blocks, NO commentary.
Generate exactly {Config.CHUNK_SIZE} objects in the array.

"""
        
        # adding JSON examples based on shot type
        if shot_type == "one":
            examples = get_examples(domain)
            if examples:
                prompt += f"EXAMPLE ROW:\n{json.dumps(examples[0], indent=2)}\n\n"
        elif shot_type == "few":
            examples = get_examples(domain)
            if examples:
                prompt += f"EXAMPLE ROWS:\n{json.dumps(examples[:3], indent=2)}\n\n"
    
    prompt += "Generate diverse, realistic data for the specified time period."
    
    if chunk_num > 1:
        prompt += f" (This is chunk {chunk_num} - generate new distinct data.)"
    
    return prompt


######### GENERATE CHUNK #########

def hf_generate(client: InferenceClient, prompt: str, model: str) -> str:
    """Generate data using HuggingFace InferenceClient"""
    try:
        if "chat" in model.lower() or "instruct" in model.lower():
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            return str(response)
        else:
            response = client.text_generation(
                prompt,
                return_full_text=False,
            )
            return response
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")


def generate_chunk(client: InferenceClient, model: str, domain: str, shot_type: str,
                   format_type: str, chunk_num: int, output_dir: Path) -> tuple[bool, Optional[pd.DataFrame], Dict]:
    '''Generate a single chunk of data.'''
    prompt = build_prompt(domain, format_type, shot_type, chunk_num)
    chunk_start_time = time.time()

    for attempt in range(1, Config.MAX_RETRIES + 1):
        error_message = None
        response_text = ""
        df = None
        schema_issues: Dict[str, Any] = {}

        try:
            print(f"    Chunk {chunk_num}, attempt {attempt}...", end=" ", flush=True)

            prompt_file = output_dir / f"chunk_{chunk_num:03d}_attempt_{attempt}_prompt.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)

            response_text = hf_generate(client, prompt, model)

            response_file = output_dir / f"chunk_{chunk_num:03d}_attempt_{attempt}_raw_response.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response_text)

            if format_type == "csv":
                df, parse_info = parse_llm_csv(response_text.strip(), domain)
                if df is not None and not df.empty:
                    schema_issues = {
                        "had_schema_mismatch": parse_info.get("schema_correction_needed", False),
                        "original_columns": parse_info.get("original_columns"),
                        "expected_columns": parse_info.get("expected_columns"),
                        "correction_type": parse_info.get("correction_type"),
                        "corrections_made": parse_info.get("corrections_made", []),
                        "header_present": parse_info.get("header_present"),
                        "used_expected_schema": parse_info.get("used_expected_schema")
                    }
            else:
                df = parse_llm_json(response_text.strip())
                df, schema_issues = validate_and_fix_schema(df, domain)

            if df is not None and len(df) > 0:
                parsed_file = output_dir / f"chunk_{chunk_num:03d}_attempt_{attempt}_parsed.{format_type}"
                if format_type == "csv":
                    df.to_csv(parsed_file, index=False)
                else:
                    df.to_json(parsed_file, orient="records", indent=2)

                if schema_issues.get("had_schema_mismatch", False):
                    schema_log_file = output_dir / f"chunk_{chunk_num:03d}_attempt_{attempt}_schema_corrections.json"
                    with open(schema_log_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "chunk_id": chunk_num,
                            "attempt": attempt,
                            "model": model,
                            "domain": domain,
                            "timestamp": datetime.now().isoformat(),
                            **schema_issues
                        }, f, indent=2)
                    print(f"{len(df)} rows (SCHEMA FIXED)")
                else:
                    print(f"{len(df)} rows")

                success = True
                break
            else:
                print("No valid data")
                success = False
                error_message = "parsing_failed_empty_output"

        except Exception as e:
            error_message = str(e)
            success = False
            print(f"Error: {str(e)[:80]}...")

        if attempt < Config.MAX_RETRIES:
            time.sleep(Config.RETRY_DELAY)

    chunk_duration = time.time() - chunk_start_time

    if success:
        return True, df, {
            "chunk_number": chunk_num,
            "total_attempts": attempt,
            "successful_attempt": attempt,
            "chunk_duration_seconds": round(chunk_duration, 2),
            "success": True,
            "final_rows": len(df),
            "final_columns": len(df.columns),
            "schema_issues": schema_issues
        }
    else:
        return False, None, {
            "chunk_number": chunk_num,
            "total_attempts": Config.MAX_RETRIES,
            "successful_attempt": None,
            "chunk_duration_seconds": round(chunk_duration, 2),
            "success": False,
            "final_rows": 0,
            "final_columns": 0,
            "error_message": error_message
        }


######### COMPUTE OUTPUT DIRECTORY #########

def _compute_output_dir(model: str, domain: str, shot_type: str, format_type: str,
                        run_tag: Optional[str], autoincrement: bool) -> Tuple[str, Path]:
    '''Compute output directory.'''
    base = Config.BASE_OUTPUT_DIR
    base_slug = Path(model.split('/')[-1].lower())
    slug = f"{base_slug}-{run_tag}" if run_tag else f"{base_slug}"
    out_dir = base / slug / domain / shot_type / format_type
    if autoincrement:
        if out_dir.exists():
            idx = 1
            while True:
                candidate_slug = f"{slug}-run{idx}"
                candidate_dir = base / candidate_slug / domain / shot_type / format_type
                if not candidate_dir.exists():
                    slug = candidate_slug
                    out_dir = candidate_dir
                    break
                idx += 1
    return slug, out_dir


######### GENERATE DATASET #########

def generate_dataset(model: str, domain: str, shot_type: str, format_type: str,
                     run_tag: Optional[str] = None, autoincrement: bool = False) -> Dict:
    if not HF_CLIENT_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    token = (os.environ.get("MY_ENV_VAR") or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF token not set. Set MY_ENV_VAR or HF_TOKEN environment variable")

    client = InferenceClient(model=model, token=token)

    used_slug, output_dir = _compute_output_dir(model, domain, shot_type, format_type, run_tag, autoincrement)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print(f"HF Generator Model: {model}")
    print(f"Domain: {domain} | Shot: {shot_type} | Format: {format_type}")
    print(f"Target: {Config.TARGET_ROWS} rows ({Config.CHUNK_SIZE} per chunk)")
    print(f"Output: {output_dir}")
    print("")

    start_time = time.time()
    chunk_logs: List[Dict[str, Any]] = []
    total_rows = 0
    failed_streak = 0
    chunk_num = 0
    all_chunks: List[pd.DataFrame] = []

    while total_rows < Config.TARGET_ROWS and failed_streak < Config.MAX_FAILED_STREAK:
        chunk_num += 1
        success, chunk_df, log_info = generate_chunk(client, model, domain, shot_type, format_type, chunk_num, output_dir)

        chunk_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "domain": domain,
            "shot_type": shot_type,
            "format_type": format_type,
            "chunk_id": chunk_num,
            "retries": log_info.get("total_attempts", 0),
            "success": success,
            "error_type": log_info.get("error_message") if not success else None,
            "rows_generated": log_info.get("final_rows", 0),
            "duration_s": log_info.get("chunk_duration_seconds", 0),
            "raw_path": f"chunk_{chunk_num:03d}_attempt_{log_info.get('successful_attempt', log_info.get('total_attempts', 0))}_raw_response.txt",
            "parsed_path": f"chunk_{chunk_num:03d}_attempt_{log_info.get('successful_attempt', 0)}_parsed.{format_type}" if success else None,
            "had_schema_mismatch": log_info.get("schema_issues", {}).get("had_schema_mismatch", False),
            "schema_correction_type": log_info.get("schema_issues", {}).get("correction_type"),
            "schema_corrections_count": len(log_info.get("schema_issues", {}).get("corrections_made", [])),
            "header_present": log_info.get("schema_issues", {}).get("header_present"),
            "used_expected_schema": log_info.get("schema_issues", {}).get("used_expected_schema"),
        }
        chunk_logs.append(chunk_entry)

        generation_log_path = output_dir / "generation_log.json"
        try:
            if generation_log_path.exists():
                existing = json.loads(generation_log_path.read_text(encoding='utf-8'))
                if isinstance(existing, list):
                    existing.append(chunk_entry)
                    generation_log_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
                else:
                    generation_log_path.write_text(json.dumps([chunk_entry], indent=2), encoding='utf-8')
            else:
                generation_log_path.write_text(json.dumps([chunk_entry], indent=2), encoding='utf-8')
        except Exception:
            pass

        if success:
            total_rows += len(chunk_df)
            failed_streak = 0
            all_chunks.append(chunk_df)
            print(f"  Progress: {total_rows}/{Config.TARGET_ROWS} rows")
        else:
            failed_streak += 1
            print(f"  Failed streak: {failed_streak}/{Config.MAX_FAILED_STREAK}")

    combined_path_str = None
    combined_rows = 0
    if all_chunks:
        combined_df = pd.concat(all_chunks, ignore_index=True)
        combined_rows = len(combined_df)
        base_name = f"{used_slug}_{domain}_{shot_type}_{format_type}"
        combined_path = output_dir / f"{base_name}.{format_type}"
        if format_type == "csv":
            combined_df.to_csv(combined_path, index=False)
        else:
            combined_df.to_json(combined_path, orient="records", indent=2)
        combined_path_str = str(combined_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "domain": domain,
        "shot_type": shot_type,
        "format_type": format_type,
        "target_rows": Config.TARGET_ROWS,
        "achieved_rows": total_rows,
        "duration_s": round(time.time() - start_time, 2),
        "chunks": chunk_logs,
        "combined_rows": combined_rows,
        "combined_path": combined_path_str,
        "run_tag": run_tag,
        "used_slug": used_slug,
        "autoincrement": autoincrement,
    }
    summary_file = output_dir / "generation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    return summary


######### RUN BATCH EXPERIMENTS #########

def run_experiment_batch(models: List[str], domains: List[str], 
                         shot_types: List[str], formats: List[str],
                         run_tag: Optional[str] = None, autoincrement: bool = False) -> List[Dict]:
    """Run multiple experiments"""


    results: List[Dict] = []
    total_experiments = len(models) * len(domains) * len(shot_types) * len(formats)
    
    print("")
    print(f"Starting batch generation: {total_experiments} experiments")
    print(f"Models: {models}")
    print(f"Domains: {domains}")
    print(f"Shot types: {shot_types}")
    print(f"Formats: {formats}")
    
    experiment_num = 0
    for model in models:
        for domain in domains:
            for shot_type in shot_types:
                for format_type in formats:
                    experiment_num += 1
                    print("")
                    print(f"[{experiment_num}/{total_experiments}] {model} | {domain} | {shot_type} | {format_type}")
                    
                    result = generate_dataset(
                        model=model,
                        domain=domain,
                        shot_type=shot_type,
                        format_type=format_type,
                        run_tag=run_tag,
                        autoincrement=autoincrement,
                    )
                    results.append(result)
                    
                    # pause between experiments
                    time.sleep(1)
    
    # saving batch summary
    summary_path = Config.BASE_OUTPUT_DIR / "batch_summary.json"
    successful = sum(1 for r in results if r.get("achieved_rows", 0) > 0)
    failed = len(results) - successful
    with open(summary_path, 'w') as f:
        json.dump({
            "total_experiments": total_experiments,
            "successful": successful,
            "failed": failed,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print("")
    print(f"Batch complete. Summary saved to {summary_path}")
    print(f"Success rate: {successful}/{total_experiments} ({(100*successful/total_experiments if total_experiments else 0):.1f}%)")
    
    return results

def generate_single(model: str, domain: str, shot_type: str = "zero", format_type: str = "csv",
                    run_tag: Optional[str] = None, autoincrement: bool = False):
    return generate_dataset(model, domain, shot_type, format_type, run_tag=run_tag, autoincrement=autoincrement)