### IMPORTS ###
import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from scipy import stats
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
import re


# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################



def find_project_root():
    current = Path.cwd()
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent
    return Path.cwd()





# Method to list all the generated datasets

# Parameters: 
# 1. Generated directory

# Returns
# 1. All the information about the datasets in a pandas dataframe. Includes domain, model, shot, run and file path.

def list_generated_datasets(generated_dir: Path):

    rows = []

    # The final version of our datasets is the one that has the string "csv_clean.csv" in the filename. 
    # so we will list all the files that have the string "csv_clean.csv" in the filename.
    csv_files = list(generated_dir.rglob("*csv_clean.csv"))
    print(f"Found {len(csv_files)} CSV files.")

    for file in csv_files:
        filename = file.stem  # remove the .csv extension

       # All our datasets in our are formatted like this: model-raw_finals_of_all_chunks_domain_shot_csv.csv
       # so we will split the filename in order to extract the domain and shot and save everything in a dataframe.
        if "_finals_of_all_chunks_" in filename:
            prefix, rest = filename.split("_finals_of_all_chunks_", 1)
        elif "_final_of_all_chunks_" in filename:
            prefix, rest = filename.split("_final_of_all_chunks_", 1)
        else:
            # We skip the files that do not match the naming pattern.
            print(f"Skipping: {filename}")
            continue

        # Extract domain and shot from the filename.
        rest_parts = rest.split("_")
        if len(rest_parts) < 4:
            print(f"Skipping (unexpected tail): {filename}")
            continue

        domain = rest_parts[0]
        shot   = rest_parts[1]

       
        m = re.search(r"(run\d+)", prefix)
        if m:
            run = m.group(1)
            model = prefix[:m.start()].rstrip("-_")
        else:
            run = None
            model = prefix

        # Save all the data.
        rows.append({
            "domain": domain,
            "model": model,
            "shot": shot,
            "run": run,
            "file_path": str(file)
        })

    # Return all the data.
    return pd.DataFrame(rows)


# Method to print a formatted table for ease of reading.

# Parameters:
# 1. df: pandas Data containing at least ['model', 'run', 'shot', value_col]
# 2. value_col: the column to summarize 
# 3. Title: optional header for the title we want to print.

# Returns:
# 1. The formatted table

def print_formatted_table(df, value_col, title="Formatted Table"):

    # Normalize run labels to "Run X"
    def normalize_run(r):
        if isinstance(r, str) and r.lower().startswith("run"):
            num = ''.join(filter(str.isdigit, r))
            return f"Run {num}"
        return r
    
    df = df.copy()
    df["run_label"] = df["run"].apply(normalize_run)
    
    # Create pivot
    pivot = df.pivot_table(
        index=["model", "run_label"],
        columns="shot",
        values=value_col,
        aggfunc="first"  
    )
    
    # Printing the formatted table.
    print(f"\n{title}:\n")
    print(pivot.to_string())
    
   
    return pivot  



# Method to load a CSV file into a DataFrame while skipping malformed or bad lines.

# Parameters:
# 1. file_path: Path to the CSV file.

# Returns: 
# 1. The loaded data

def load_csv_safely(file_path):

    try:
        df = pd.read_csv(
            file_path,
            on_bad_lines="skip",   
            engine="python",       
            dtype=str              
        )
        return df

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame() 


# Method to print the unique values for each sensitive attribute, for the real and LLM generated datasets

# Parameters:
# 1. SENSITIVE_ATTRIBUTES: Dictionary containing the sensitive attributes for each domain
# 2. real_data_by_domain: Dictionary containing the real datasets for each domain
# 3. llm_data_by_domain: Dictionary containing the LLM generated datasets for each domain

# Returns:
# 1. The unique values for each sensitive attribute, for the real and LLM generated datasets
def print_unique_sensitive_values(SENSITIVE_ATTRIBUTES, real_data_by_domain, llm_data_by_domain):
    
    for domain, attributes in SENSITIVE_ATTRIBUTES.items():
        print("=" * 80)
        print(f"DOMAIN: {domain.upper()}")
        print("=" * 80)

        real_df = real_data_by_domain[domain]
        llm_df  = llm_data_by_domain[domain]

        for attr in attributes:
            print(f"\nSensitive Attribute: {attr}")

            # Unique values (REAL)
            real_vals = sorted(real_df[attr].dropna().unique().tolist()) \
                        if attr in real_df.columns else "Not in real dataset"

            # Unique values (LLM — all combined)
            llm_vals = sorted(llm_df[attr].dropna().unique().tolist()) \
                       if attr in llm_df.columns else "Not in LLM datasets"

            print(f"  • Real unique values ({len(real_vals) if isinstance(real_vals, list) else '-' }):")
            print(f"      {real_vals}")

            print(f"  • LLM unique values ({len(llm_vals) if isinstance(llm_vals, list) else '-' }):")
            print(f"      {llm_vals}")

        print("\n")


# Dictionary used for the mapping of the LLM generated datasets to the real datasets.
ATTRIBUTE_ALIASES = {
    "lending": {
        "derived_ethnicity": {
            "Hispanic or Latino": [

                "hispanic or latino",
                "hispanic",
                "hisp or latino",
                "hisp. or latino",
                "hispanic white",
                "hispanic or latino - white",
                "hispanic/mexican",
                "hispanic/latino",

                
                "latino",
                "latinos",
                "mexican",
                "central american",
                "south american",
                "other latino",
                "hispatic or latino",
                "hispidian or latino",  # from HISPIDIAN OR LATINO
                "husban hispanic or latino",

                # quoted / escaped variants once normalized
                "hispanic or latino",
            ],

            "Not Hispanic or Latino": [
                "not hispanic or latino",
                "not hispanic",
                "non-hispanic",
                "non hispanic",
                "non-his",
                "non-hispanic white",
                "non-hispanic white - white",
                "non-hispanic or latino",
                "non-hispanic asian",
                "non-hispanic black or african american",
                "non hispanic",
                "other non-hispanic or latino",
                "not hispanic or other ethnicity",


                "non-hispanic",
                "not hispanic or latino",
            ],

            "Ethnicity Not Available": [
                "ethnicity not available",
                "history of ethnicity not available",
                "unknown",
                "not reported",
                "information not provided by applicant in mail",
                "other ethnicity",
                "other", 

            ],

            "Free Form Text Only": [
                "free form text only",
            ],

            "Joint": [
                "joint",
            ],
        },


        "derived_race": {
            "White": [
                "white",
                "white male",
                "white female",
                "caucasian",
                "white or caucasian",
                "white/unknown",
                "non-hispanic white",
                "hispanic white",   
            ],

            "Black or African American": [
                "black",
                "african american",
                "black or african american",
                "black or african american female",
                "black or african american male",
                "black/african american",
            ],

            "Asian": [
                "asian",
                "asians",
                "asian female",
                "asian male",
                "asian or pacific islander",   
                "asian/pacific islander",
                "other asian",
                "yellow/asian",
                
            ],

            "American Indian or Alaska Native": [
                "american indian",
                "alaskan native",
                "american indian or alaska native",
                "american indian/alaska native",
                "native american",
                "native american or alaska native",
                "native american/ alaska native",
                "native american/alascan",
                "indian", 
            ],

            "Native Hawaiian or Other Pacific Islander": [
                "native hawaiian or other pacific islander",
                "native hawaiian or pacific islander",
                "hawaiian native or other pacific islander",
                "hawaiian or pacific islander",
                "native hawaiian",
                "pacific islander",
                "islander",
                "asian/native hawaiian/other pacific islander",
            ],

            "2 or more minority races": [
                "2 or more minority races",
                "mixed race",
                "multi-racial",
                "multiracial",
                "mixed/multiple races",
                "other/mixed",
                "other/foreign",
                "others",
                "biasian", 
            ],

            "Race Not Available": [
                "race not available",
                "unknown",
                "not reported",
                "not specified",
                "some other race",
                "other race",
                "other",         
                "sex not available",  
            ],

            "Free Form Text Only": [
                "free form text only",
            ],

            "Joint": [
                "joint",
                "both",
                "couple",
            ],
        },

        "derived_sex": {
            "Female": [
                "female",
                "b female",
                "binary-female",
                "other/female",
                "female ",  
                
            ],

            "Male": [
                "male",
                "binary-male",
                "male ",   
            ],

            "Sex Not Available": [
                "sex not available",
                "unknown",
                "unkown",
                "not specified",
                "not applicable",
                "neither",
                "gender non-conforming",
                "non-binary",
                "non-binary",
                "non-binary",  
                "prefers not to identify",
                "other sex",
                "other",       
            ],

            "Joint": [
                "joint",
                "both",
                "couple",
            ],
        },
    },

    "hatecrime": {
        "offender_race": {
            "White": [
                "white",
                "white or caucasian",
                "white/unknown",
                "european american",
                "w",
                "white ",
                "white/caucasian",
            ],

            "Black or African American": [
                "black",
                "african american",
                "african american or black",
                "african american/black",
                "black or african american",
                "black or african native",
                "black/african american",
                "b",
                "black ",
            ],

            "Asian": [
                "asian",
                "asain",
                "asiian",
                "asian american",
                "asian or asian american",
                "asian or asian/pacific islander",
                "asian or pacific islander",
                "asian/pacific islander",
                "asian-american",
                "asiatic/pacific islander",
                "asians",
                "yellow/asian",
                "a",
                "asian/other",
            ],

            "American Indian or Alaska Native": [
                "american indian",
                "american indian or alaska native",
                "american indian/alaska native",
                "american indian/alaskan native",
                "indian",
                "indian or alaska native",
                "indian/alaskan",
                "native american",
                "native american or alaska native",
                "native american indian or alaska native",
                "native american or american indian",
                "native american/alaska native",
                "native american/alaskan",
                "native american/native alaskan",
                "alaskan native",
            ],

            "Native Hawaiian or Other Pacific Islander": [
                "native hawaiian or other pacific islander",
                "native hawaiian or pacific islander",
                "native hawaiian/other pacific islander",
                "hawaiian or other pacific islander",
                "hawaiian or pacific islander",
                "pacific islander",
                "asian/native hawaiian/other pacific islander",
                "native hawaiian",
                "islander",
            ],

            "Multiple": [
                "multiple",
                "multiple races",
                "mixed race",
                "mixed/multiple races",
                "mixed",
                "mixed group",
                "multiracial",
                "other or multiple",
                "w/h",
                "w/h/b",
            ],

            "Not Specified": [
                "not specified",
                "not_specified",
                "non-specific",
                "offender race",      
                "offender_race",
            ],

            "Unknown": [
                "unknown",
                "unknown/unknown",
                "ethnicity unknown",
                "offender_race",  
            ],
        },

        "offender_ethnicity": {
            "Hispanic or Latino": [
                "hispanic or latino",
                "hispanic",
                "hisp/latino",
                "hisp/other",         # better here than non-hisp
                "hisp/chicano",
                "histpanic",        
                "hispanic/latino",
                "hispanic/mexican",
                "hispanic/mexican/chicano",
                "hispanic/puerto rican",
                "latino",
                "hispanic or latino", 
            ],

            "Not Hispanic or Latino": [
                "not hispanic or latino",
                "not hispanic",
                "non-hispanic",
                "non hispanic",
                "non hispanic",   
                "non hispanic or latino",
                "not hispanic or other ethnicity",
                "non hispanic",
                "non-hispanic",
            ],

            "Not Specified": [
                "not specified",
                "not specify",
                "not_specified",
            ],

            "Unknown": [
                "unknown",
            ],

            "Multiple": [
                "mixed race",   
                "multiple",
            ],
        },
    },
}



# Method to normalize the strings for matching (case-insensitive, trims quotes/spaces).

# Parameters:
# 1. x: The string to normalize

# Returns:
# 1. The normalized string

def _norm_val(x):
    """Normalize strings for matching (case-insensitive, trims quotes/spaces)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.strip("\"'“”‘’")          # strip surrounding quotes
    s = s.lower()                   # lowercase
    s = re.sub(r"\s+", " ", s)      # collapse multiple spaces
    return s




# Another method used to normalize the strings for matching (case-insensitive, trims quotes/spaces). Used later for
# the outcome attributes.

# Parameters:
# 1. x: The string to normalize

# Returns:
# 1. The normalized string

def normalize_outcome_value(x):
    """Lowercase + strip + remove quotes + collapse whitespace."""
    if isinstance(x, str):
        x = x.strip().lower()
        x = x.replace('"', "").replace("'", "")
        x = " ".join(x.split())
    return x





# Method to apply the mappings to all the datasets in the metadata table

# Parameters:
# 1. metadata_table: The metadata table containing the datasets
# 2. ATTRIBUTE_ALIASES: The dictionary containing the attribute aliases

# Returns:
# 1. The metadata table with the mapped datasets

def apply_mappings_to_all_datasets(metadata_table, ATTRIBUTE_ALIASES):

    for idx, row in metadata_table.iterrows():
        domain = row["domain"]
        df = row["data"]

        if df is None or not isinstance(df, pd.DataFrame):
            continue

        domain_aliases = ATTRIBUTE_ALIASES.get(domain, {})
        if not domain_aliases:
            continue

        for attr, mapping_by_real in domain_aliases.items():

            if attr not in df.columns:
                continue

            raw_to_real = {}

            for real_value, aliases in mapping_by_real.items():
                # map canonical label to itself
                canon_norm = _norm_val(real_value)
                if canon_norm is not None:
                    raw_to_real[canon_norm] = real_value

                # map each alias to canonical label
                for alias in aliases:
                    alias_norm = _norm_val(alias)
                    if alias_norm is not None:
                        raw_to_real[alias_norm] = real_value


            new_col = f"{attr}_Mapped"
            df[new_col] = df[attr].apply(
                lambda x: raw_to_real.get(_norm_val(x), np.nan)
            )


        metadata_table.at[idx, "data"] = df




# Method to count the number of NaN values in the mapped datasets

# Parameters:
# 1. metadata_table: The metadata table containing the datasets
# 2. ATTRIBUTE_ALIASES: The dictionary containing the attribute aliases

# Returns:
# 1. The number of NaN values in the mapped datasets
def find_bad_mappings_for_config(
    metadata_table,
    SENSITIVE_ATTRIBUTES,
    domain,
    model=None,
    run=None,
    shot=None,
    threshold=None,
):

    # Build mask to select the specific dataset
    mask = metadata_table["domain"] == domain
    if model is not None:
        mask &= metadata_table["model"] == model
    if run is not None:
        mask &= metadata_table["run"] == run
    if shot is not None:
        mask &= metadata_table["shot"] == shot

    subset = metadata_table[mask]

    if subset.empty:
        print("No dataset found for the given filters.")
        return pd.DataFrame()

    # If more than one matches, we’ll just check all of them,
    # but usually you expect exactly one.
    
    results = []

    for _, row in subset.iterrows():
        df = row["data"]
        if df is None:
            continue

        this_model = row["model"]
        this_run   = row["run"]
        this_shot  = row["shot"]

        attrs = SENSITIVE_ATTRIBUTES.get(domain, [])
        for attr in attrs:
            mapped_col = f"{attr}_Mapped"
            if mapped_col not in df.columns:
                continue

            total_rows = len(df[mapped_col])
            nan_count = df[mapped_col].isna().sum()
            if total_rows == 0:
                continue

            nan_ratio = nan_count / total_rows

            if nan_ratio > threshold:
                results.append({
                    "domain": domain,
                    "model": this_model,
                    "run": this_run,
                    "shot": this_shot,
                    "attribute": attr,
                    "nan_count": nan_count,
                    "total_rows": total_rows,
                    "nan_pct": nan_ratio * 100,
                })

    bad_df = pd.DataFrame(results)
    if bad_df.empty:
        print(f"No mappings above {threshold*100:.0f}% NaN for this config.")
        return bad_df

    bad_df = bad_df.sort_values(by="nan_pct", ascending=False)

    print(f"Mappings with more than {threshold*100:.0f}% NaN for this dataset:\n")
    print(bad_df.to_string(index=False, formatters={"nan_pct": "{:.2f}".format}))
    return bad_df




# Method to print the unique values for each outcome attribute, for the real and LLM generated datasets

# Parameters:
# 1. OUTCOME_ATTRIBUTES: The dictionary containing the outcome attributes
# 2. real_data_by_domain: The dictionary containing the real datasets
# 3. llm_data_by_domain: The dictionary containing the LLM generated datasets

# Returns:

def print_unique_outcome_values(OUTCOME_ATTRIBUTES, real_data_by_domain, llm_data_by_domain):

    for domain, attributes in OUTCOME_ATTRIBUTES.items():
        print("=" * 80)
        print(f"DOMAIN: {domain.upper()}")
        print("=" * 80)

        real_df = real_data_by_domain[domain]
        llm_df = llm_data_by_domain[domain]

        for attr in attributes:
            print(f"\nOutcome Attribute: {attr}")

            # Real uniques
            if attr in real_df.columns:
                real_vals = real_df[attr].dropna().unique().tolist()
            else:
                real_vals = "Not in real dataset"

            # LLM uniques
            if attr in llm_df.columns:
                llm_vals = llm_df[attr].dropna().unique().tolist()
            else:
                llm_vals = "Not in LLM datasets"

            print(f"  • Real unique values ({len(real_vals) if isinstance(real_vals, list) else '-' }):")
            print(f"      {real_vals}")

            print(f"  • LLM unique values ({len(llm_vals) if isinstance(llm_vals, list) else '-' }):")
            print(f"      {llm_vals}")

        print("\n")




# Dictionary for the mapping of outcome values.
OUTCOME_ALIASES = {
    "lending": {
        # Map action_taken (real + LLM) -> loan_approved (1 = approved/originated, 0 = not approved)
        "action_taken": {
            1: [
                # clear approvals / originations
                "loan originated",
                "loan originations",
                "originated",
                "made a loan",
                "made loan",
                "loan made",
                "loan consummated",
                "loan funded",
                "funded",
                "approved and funded",
                "loan approved and funded",
                "loan approved and funds disbursed",
                "loan approved and submitted",
                "loan approved and submitted to underwriting",
                "loan approved after submission",
                "approved and accepted",
                "approved and funded",
                "completed",
                "loan closed", 
                "loan finalized for disbursement",
                "loan consummated",
       
                "1", "1.0",
            ],
            0: [
                # clear denials
                "application denied",
                "denied",
                "loan denied",
                "denial",
                "denied by financial institution",
                "denial of loan application",
                "denied due to credit history",
                "denied due to insufficient income",
                "denied due to insufficient credit",
                "loan rejected",
                "loan denid",
                "loan denial",
                "denied application",
                "deny",
                "deny application",
                "deny by institution",
                # withdrawn / incomplete
                "application withdrawn",
                "application withdraw",
                "application withdrawn by applicant",
                "application withdrew",
                "application withdrew by applicant",
                "file closed for incompleteness",
                "file closed",
                "loan closed for incompleteness",
                # approved but not accepted → treat as not approved (no loan)
                "application approved but not accepted",
                "loan approved but not accepted",
                "approved but not accepted",
                "approved but not accepted by applicant",
                "approved not accepted",
                # preapproval denials / not materialized
                "preapproval request denied",
                "preapproval denied",
                "preapproval request approved but not accepted",
                "preapproval request cancelled",
  
                "loan not made",
                "loan not accepted",
                "no approval",
                "loan not applied for",

                "2", "2.0", "3", "3.0", "4", "4.0", "5", "5.0",
                "6", "6.0", "7", "7.0", "8", "8.0",
            ],
        },
    },

    "hatecrime": {
       
        "offense_name": {
            "violent": [
                # core violent offenses
                "aggravated assault",
                "simple assault",
                "assault",
                "felony assault",
                "felonious assault",
                "battery",
                "attack",
                "attack/assault",
                "physical attack",
                "bodily injury from assault",
                "bodily harm",

                "robbery",
                "armed robbery",
                "robbery;simple assault",

                "murder",
                "homicide",
                "murder and nonnegligent manslaughter",
                "murder & non-negligent manslaughter",
                "murder/non-negligent manslaughter",
                "murder and nonnegligent homicide",
                "attempted murder",
                "manslaughter",

                "rape",
                "forcible rape",
                "sexual assault",
                "sexual abuse",
                "sexual abuse/assault",
                "criminal sexual contact",
                "sexual assault with an object",
                "sodomy",
                "forcible fondling",
                "statutory rape",
                "molesting",

                "kidnapping",
                "kidnapping/abduction",
                "abduction/kidnapping",
                "abduction/unlawful restraint",

                "shooting",
                "assault with a weapon",
                "assault with a dangerous weapon",
                "assault with firearm",
                "attack with firearm",
                "assault with serious bodily injury",
                "assault with intent to commit murder",
                "assault with an automatic weapon",

                # generic violent labels
                "violent crime",
                "hate crime murder",
            ],

            "non-violent": [
                # property crimes
                "destruction/damage/vandalism of property",
                "damage/destruction/vandalism of property",
                "destruction/vandalism",
                "vandalism",
                "criminal mischief",
                "mischief",
                "property damage",

                "larceny-theft",
                "larceny theft",
                "larceny/theft",
                "theft",
                "petty theft",
                "shoplifting",
                "stolen property offenses",
                "theft from motor vehicle",
                "theft from building",
                "theft of motor vehicle parts or accessories",

                "burglary",
                "burglary/breaking & entering",
                "burglary/breaking and entering",
                "housebreaking",

                "motor vehicle theft",

                "arson",

                # drug / weapons / other non-personal
                "drug/narcotic violations",
                "drug abuse violations",
                "drug equipment violations",
                "weapon law violations",
                "possession of weapon",
                "possession of firearm",

                # fraud-type
                "fraud",
                "false pretenses/swindle/confidence game",
                "counterfeiting/forgery",
                "embezzlement",
                "credit card/automated teller machine fraud",
                "identity theft",
                "wire fraud",
                "hacking/computer invasion",

             
                "intimidation",
                "harassment",
                "harassment/intimidation",
                "threat",
                "threats",
                "threatening words",
                "criminal threats",
                "fear/intimidation",
                "bias intimidation",

      
                "hate crime",
                "hate crime incident",
                "crime against property",
                "crime against persons",  
                "other",
                "unspecified",
            ],
        },
    },
}




def build_outcome_mapping(alias_dict):
    """
    Convert {"canonical": [alias1, alias2, ...], ...}
    into a raw_value → canonical lookup dict.
    """
    mapping = {}
    for canonical, aliases in alias_dict.items():
        # canonical value maps to itself
        mapping[normalize_outcome_value(canonical)] = canonical
        # each alias maps to the canonical category
        for a in aliases:
            mapping[normalize_outcome_value(a)] = canonical
    return mapping




# Method to apply the mappings to all the datasets in the metadata table

# Parameters:
# 1. metadata_table: The metadata table containing the datasets
# 2. OUTCOME_ALIASES: The dictionary containing the outcome attributes

# Returns:
# 1. The metadata table with the mapped datasets

def _outcome_target_name(domain, outcome_attr):
    """Decide the name of the new mapped column."""
    if domain == "lending" and outcome_attr == "action_taken":
        return "loan_approved"
    if domain == "hatecrime" and outcome_attr == "offense_name":
        return "is_violent"
    return outcome_attr + "_Mapped"




def apply_outcome_mappings_all(real_data_by_domain, metadata_table, OUTCOME_ALIASES):

    outcome_mappings = {}
    for domain, attrs in OUTCOME_ALIASES.items():
        for outcome_attr, alias_dict in attrs.items():
            outcome_mappings[(domain, outcome_attr)] = build_outcome_mapping(alias_dict)


    for (domain, outcome_attr), raw_to_canonical in outcome_mappings.items():
        if domain not in real_data_by_domain:
            continue
        real_df = real_data_by_domain[domain]
        if outcome_attr not in real_df.columns:
            continue

        new_col = _outcome_target_name(domain, outcome_attr)
        real_df[new_col] = real_df[outcome_attr].apply(
            lambda x: raw_to_canonical.get(normalize_outcome_value(x), np.nan)
        )
        real_data_by_domain[domain] = real_df  # write back (optional, mostly for clarity)

    for idx, row in metadata_table.iterrows():
        domain = row["domain"]
        df = row["data"]
        if df is None:
            continue

        domain_aliases = OUTCOME_ALIASES.get(domain, {})
        if not domain_aliases:
            continue

        for outcome_attr in domain_aliases.keys():
            key = (domain, outcome_attr)
            raw_to_canonical = outcome_mappings.get(key)
            if raw_to_canonical is None or outcome_attr not in df.columns:
                continue

            new_col = _outcome_target_name(domain, outcome_attr)
            df[new_col] = df[outcome_attr].apply(
                lambda x: raw_to_canonical.get(normalize_outcome_value(x), np.nan)
            )

        metadata_table.at[idx, "data"] = df

    print("Outcome columns created on all REAL and LLM datasets successfully.")



# Method to calculate the base rate parity for the Employment dataset

# Parameters:
# 1. df: The dataset to calculate the base rate parity for
# 2. pairs: The pairs of attributes to calculate the base rate parity for

# Returns:
# 1. The base rate parity for the dataset

def base_rate_parity_employment(df, pairs):

    bp_values = []

    for male_col, female_col in pairs:
        # Drop rows with missing values in either column
        valid = df[[male_col, female_col]].dropna()

        # Positive outcome for each group
        p_male   = (valid[male_col] > valid[female_col]).mean()
        p_female = (valid[female_col] > valid[male_col]).mean()

        # Base rate parity for this pair
        bp = abs(p_male - p_female)
        bp_values.append(bp)

    # Average base rate parity across all pairs
    bp_avg = float(np.mean(bp_values)) if bp_values else np.nan

    return bp_avg, bp_values



# Method to calculate the disparate impact for the Employment dataset

# Parameters:
# 1. df: The dataset to calculate the disparate impact for
# 2. pairs: The pairs of attributes to calculate the disparate impact for

# Returns:
# 1. The disparate impact for the dataset

def disparate_impact_employment(df, pairs):

    di_values = []


    for male_col, female_col in pairs:
        df[male_col] = pd.to_numeric(df[male_col], errors="coerce")
        df[female_col] = pd.to_numeric(df[female_col], errors="coerce")


    for male_col, female_col in pairs:

        male_prob   = df[male_col].mean() / 100
        female_prob = df[female_col].mean() / 100

        if female_prob == 0 or np.isnan(female_prob):
            di = np.nan
        else:
            di = male_prob / female_prob

        di_values.append(di)

    di_avg = np.nanmean(di_values)
    return di_avg, di_values




# Method to calculate the mean difference for the Employment dataset

# Parameters:
# 1. df: The dataset to calculate the mean difference for
# 2. pairs: The pairs of attributes to calculate the mean difference for

# Returns:
# 1. The mean difference for the dataset

def mean_difference_employment(df, pairs):


    # --- CLEANING STEP (same idea as DI) ---
    for male_col, female_col in pairs:
        df[male_col]   = pd.to_numeric(df[male_col], errors="coerce")
        df[female_col] = pd.to_numeric(df[female_col], errors="coerce")
    # ---------------------------------------

    md_values = []

    for male_col, female_col in pairs:
        male_mean   = df[male_col].mean()
        female_mean = df[female_col].mean()

        if np.isnan(male_mean) or np.isnan(female_mean):
            md = np.nan
        else:
            md = abs(male_mean - female_mean)   # exactly as in the formula

        md_values.append(md)

    md_avg = np.nanmean(md_values)
    return md_avg, md_values



# Method to calculate the base rate parity for the multiclass dataset
# Parameters:
# 1. df: The dataset to calculate the base rate parity for
# 2. sensitive_attrs: The list of sensitive attributes to calculate the base rate parity for
# 3. outcome_col: The name of the outcome column
# 4. positive_label: The value of the positive label

# Returns:
# 1. The base rate parity for the dataset
def base_rate_parity(df, sensitive_attrs, outcome_col, positive_label):


    per_attr_bp = {}
    per_attr_probs = {}

    for S in sensitive_attrs:
        # Keep only rows where S and Y are not NaN
        sub = df[[S, outcome_col]].dropna()

        if sub.empty:
            per_attr_bp[S] = np.nan
            per_attr_probs[S] = pd.Series(dtype=float)
            continue

        # P(Y=1 | S=g) for each group g
        probs = sub.groupby(S)[outcome_col].apply(
            lambda s: (s == positive_label).mean()
        )

        per_attr_probs[S] = probs

        # Need at least 2 groups to form pairwise disparities
        if len(probs) < 2:
            per_attr_bp[S] = np.nan
            continue

        # Compute max pairwise disparity
        vals = probs.values
        max_disp = 0.0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                d = abs(vals[i] - vals[j])
                if d > max_disp:
                    max_disp = d

        per_attr_bp[S] = max_disp

    # Overall base rate parity = mean over sensitive attributes
    overall_bp = np.nanmean(list(per_attr_bp.values()))

    return overall_bp, per_attr_bp, per_attr_probs




# Method to calculate the disparate impact for the multiclass dataset
# Parameters:
# 1. df: The dataset to calculate the disparate impact for
# 2. sensitive_attrs: The list of sensitive attributes to calculate the disparate impact for
# 3. outcome_col: The name of the outcome column
# 4. positive_label: The value of the positive label

# Returns:
# 1. The disparate impact for the dataset
def disparate_impact_multiclass(df, sensitive_attrs, outcome_col, positive_label):
 
    per_attr_di = {}
    per_attr_probs = {}

    for S in sensitive_attrs:
        # Drop rows with missing values in S or Y
        sub = df[[S, outcome_col]].dropna()

        if sub.empty:
            per_attr_di[S] = np.nan
            per_attr_probs[S] = pd.Series(dtype=float)
            continue

        # P(Y=1 | S=g) for each group g
        probs = sub.groupby(S)[outcome_col].apply(
            lambda s: (s == positive_label).mean()
        )
        per_attr_probs[S] = probs

        vals = probs.values
        if len(vals) < 2:
            per_attr_di[S] = np.nan
            continue

        deviations = []

        # Ordered pairs (gi, gj), i != j
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i == j:
                    continue
                p_i, p_j = vals[i], vals[j]

                # skip invalid or zero-denominator cases
                if np.isnan(p_i) or np.isnan(p_j) or p_j == 0:
                    continue

                di_ij = p_i / p_j
                deviations.append(abs(di_ij - 1.0))

        per_attr_di[S] = max(deviations) if deviations else np.nan

    overall_di = np.nanmean(list(per_attr_di.values()))

    return overall_di, per_attr_di, per_attr_probs




# Method to calculate the base rate for the multiclass dataset
# Parameters:
# 1. df: The dataset to calculate the base rate for
# 2. sensitive_attrs: The list of sensitive attributes to calculate the base rate for
# 3. outcome_col: The name of the outcome column
# 4. positive_label: The value of the positive label

# Returns:
# 1. The base rate for the dataset 
def base_rate_multiclass(df, sensitive_attrs, outcome_col, positive_label):

    per_attr_br = {}
    per_attr_rates = {}

    for S in sensitive_attrs:
        # Keep only rows where S and Y are not NaN
        sub = df[[S, outcome_col]].dropna()
        if sub.empty:
            per_attr_br[S] = np.nan
            per_attr_rates[S] = pd.Series(dtype=float)
            continue

        N = len(sub)

        # Count positives per group
        pos = sub[sub[outcome_col] == positive_label].groupby(S).size()

        # Ensure all groups appear (groups with 0 positives get BR=0)
        groups = sub[S].unique()
        base_rates = pos.reindex(groups, fill_value=0) / N

        per_attr_rates[S] = base_rates

        vals = base_rates.values
        if len(vals) < 2:
            per_attr_br[S] = np.nan
            continue

        # Compute max pairwise absolute difference in base rates
        max_disp = 0.0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                d = abs(vals[i] - vals[j])
                if d > max_disp:
                    max_disp = d

        per_attr_br[S] = max_disp

    overall_br = np.nanmean(list(per_attr_br.values()))
    return overall_br, per_attr_br, per_attr_rates


def format_domain_name(domain: str) -> str:
    if domain.lower() == 'hatecrime':
        return 'Hate Crime'
    return domain.title()

