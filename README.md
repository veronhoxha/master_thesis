# Good or Bad: LLM-Generated Datasets

## Introduction
This repository hosts all resources for the master's thesis project **“Good or Bad: LLM-Generated Datasets.”** It covers everything from raw real datasets, preprocessing steps, and Hugging Face–based data generation scripts from open-soruce LLM's to the analysis about the quality, consistency, structure and completness, and bias analysis of the generated datasets compared to the real ones.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
  - [Code](#code)
  - [Data](#data)
    - [Metadata](#metadata)
    - [Subfolders](#subfolders)
  - [Additional Folders](#additional-folders)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Project Structure

### Code

All code is avaliable in the `notebooks/` folder and is organized by workflow stage:

#### Data Generation
- `notebooks/data_generation/generating_datasets.py` – Hugging Face Inference client that helps generate data in chunks through an API. In this script we do the prompt construction, normalizes CSV schemas, and batch experiments across models/domains/shots.
- `notebooks/data_generation/generating_datasets.ipynb` – Notebook for generating datasets in runs.
- `notebooks/data_generation/combine_raw_chunks_to_csv.py` – Script to merge chunk outputs from a run into final CSVs.

#### Preprocessing
- `notebooks/preprocessing/lending_preprocess.ipynb` – Cleans the lending dataset and saves the cleaned CSV datasets.
- `notebooks/preprocessing/hatecrime_preprocess.ipynb` – Cleans the hate crime dataset and saves the cleaned CSV datasets.
- `notebooks/preprocessing/genderpaygap_preprocess.ipynb` – Cleans the employment dataset and saves the cleaned CSV datasets.
- `notebooks/preprocessing/file_format_converter_csv_to_parquet.py` / `file_format_converter_csv_to_json.py` – Converters for in different formats.
- `notebooks/preprocessing/raw_and_preprocessed_data_samples.py` – Getting samples from both the raw and cleaned datasets for documentation purposes in GitHub.

#### Core Experiments & Analyses
- `notebooks/experiments/malformed_rows/analyze_bad_lines.py` – Script that inspects every generated CSV, enumerates through malformed rows, and writes the analysis under `analysis/bad_lines/`.
- `notebooks/experiments/malformed_rows/visualize_bad_lines.ipynb` / `visualize_bad_lines.py` – Plotting helpers for turning the malfomed-lines into figures.
- `notebooks/experiments/basic_details_analysis.ipynb` / `basic_details_analysis.py` – Produce statistics analysis (row counts, time progression, chunk summaries) for each generation run.
- `notebooks/experiments/clean_raw_finals.py` – Script that and produces the `_clean.csv` files of generated datasets used for experiments.
- `notebooks/experiments/color_palette.py` – Color palette to keep thesis figures consistent.
- `notebooks/experiments/create_ground_truth.py` – Builds schema ground truth of the real datasets used in some experiments.
- `notebooks/experiments/data_loading.py` – Shared helper functions for discovering `(model, domain, shot)` groups and reading CSVs.
- `notebooks/experiments/dataset_quality_checks.py` – Validates generated data against the real datasets (duplicate rates, missing value rates, logical constraints, data types) and exports summaries to `analysis/quality_checks/`.
- `notebooks/experiments/dataset_quality_visualization.ipynb` – Visuals for data quality checks.
- `notebooks/experiments/malformed_rows/validate_with_duckdb.py` – Executes DuckDB SQL validation for every cleaned CSV to confirm schema conformance compared to Pandas parser.
- `notebooks/experiments/malformed_rows/duckdb_vs_pandas_comparison.py` – Compares DuckDB row counts with pandas row counts to verify results.
- `notebooks/experiments/consistency_analysis.ipynb` – Main notebook for measuring numeric/categorical/semantic consistency across runs.
- `notebooks/experiments/consistency_metrics.py` – Library of reusable functions of stability metrics.
- `notebooks/experiments/bias/functions.py` – Core bias functions (disparate impact, base-rate parity).
- `notebooks/experiments/bias/bias.ipynb` – Main bias notebook where we identify the results.
- `notebooks/experiments/bias/plots.ipynb` – Visualization plots for bias metrics results.

### Data

All datasets are in the `data/` folder.

#### Metadata
- `data/raw/README.md` documents the public sources:
  - FBI Crime Data Explorer (hate crime)
  - UK Government Gender Pay Gap Service (employment)
  - CFPB HMDA Data Browser 2024 (lending)
- `data/preprocessed/column_type_ground_truth.json` captures the expected column types per real dataset for some metrics calculations.

#### Subfolders
- `data/raw/`
  - Contains the real datasets in CSV format (ignored in Git due to size) and lightweight `sample_*.{csv}` files to have some example data in GitHub.
- `data/preprocessed/`
  - Cleaned versions of each dataset plus sample subsets, used as the “real data” baseline.
- `data/generated/`
  - Outputs produced by the LLM data generator script organized as `data/generated/<model>-runX/<domain>/<shot>/<format>/`. Each run stores raw responses, parsed CSVs, combined CSV file, logs, and generation summary in JSON.

### Additional Folders
- `analysis/`
  - `basic_details/` – Row counts, time progression, chunk summaries.
  - `bad_lines/` – Malformed rows analysis.
  - `duckdb_validation/` – DuckDB vs Pandas malformed rows validation logs and comparisons.
  - `quality_checks/` – Real-vs-generated constraint summaries.
  - `consistency/` – Consistency metrics for generated datasets per model/domain/shot.
  - `bias/` – Bias metrics per domain.
- `figures/` – Final figures (flow charts, metric plots) used in the thesis.
- `.env` – Environment variables (e.g., `MY_ENV_VAR` / `HF_TOKEN`) required for the Hugging Face data generator. This file is git-ignored, create `.env` file in the main directory and add an API key from Hugging Face if needed to generate data.

## Installation

Prerequisite: Python **3.13** (project developed/tested with 3.13.x, other versions may work but are unverified).

1. **Create and activate a virtual environment**
   - macOS/Linux  
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows  
     ```powershell
     python3 -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install required packages**  

    When the virtual environment is activated, install all necessary packages by running:
    - `pip install -r requirements.txt`

3. **Configure secrets**  
   ```bash
   echo 'MY_ENV_VAR="hf_xxx_your_token"' > .env
   ```
   Keep `.env` private.

4. **Populate `data/raw/`**  
   Download the full datasets referenced in below if you plan to regenerate preprocessed files and re-run the analyses. Sample files alone are not sufficient for the complete experiments.
   Real dataset names:  `hate_crime.csv`, `uk_gender_pay_gap_data_2024_to_2025`, `year_2024.csv` (please use same namings when downloading the raw datasets).

  - Source of data for hate crime: https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads

  - Source of the data for employment: https://gender-pay-gap.service.gov.uk/viewing/download

  - Source of data for lending: https://ffiec.cfpb.gov/data-browser/data/2024?category=nationwide



## Usage
To reproduce the full thesis pipeline:

1. **Preprocess reference datasets**  
   Execute the notebooks in `notebooks/preprocessing/` after generating the raw datasets(`lending_preprocess.ipynb`, `hatecrime_preprocess.ipynb`, `genderpaygap_preprocess.ipynb`) to create updated CSV files in `data/preprocessed/`.

2. **Generate LLM datasets**  
   Run `notebooks/data_generation/generating_datasets.ipynb`.
   Verify that `data/generated/` now contains the run folders.

3. **Run Experiements**  
    Execute the files under the `notebooks/experiments/` folder.

## Contributors
- **Veron Hoxha** . Contact: `veho@itu.dk` / `veronhoxha@yahoo.com`
- **Estref Katillari** . Contact: `ekat@itu.dk` / `estrefkatillari00@gmail.com`

## License
This project is licensed under the MIT License.