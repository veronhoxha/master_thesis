"""
Clean Raw Finals CSV Files

This script loads raw_finals CSV files using pandas with error handling,
removes malformed rows, and saves cleaned versions that are guaranteed
to be readable by pandas.

For each file: `*-raw_finals_of_all_chunks_*.csv`
Creates: `*-raw_finals_of_all_chunks_*_clean.csv` in the same directory.
"""

### IMPORTS ###

import pandas as pd
from pathlib import Path
import sys


# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

###########################



def find_project_root():
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent
    return Path.cwd()


def discover_raw_finals_files(generated_root: Path):
    """
    Find all raw_finals CSV files.
    
    Returns list of CSV file paths.
    """
    
    csv_files = []
    for csv_file in generated_root.rglob("*raw_finals_of_all_chunks*.csv"):
        if "_clean.csv" not in csv_file.name:  
            csv_files.append(csv_file)
    
    return sorted(csv_files)


def clean_csv_file(csv_path: Path) -> dict:
    """
    Clean a CSV file by removing malformed rows.
    
    Returns:
        {
            'original_rows': int,
            'cleaned_rows': int,
            'rows_dropped': int,
            'schema': dict,
            'error': str or None
        }
    """
    
    result = {
        'original_rows': 0,
        'cleaned_rows': 0,
        'rows_dropped': 0,
        'schema': {},
        'error': None
    }
    
    try:
        try:
            df = pd.read_csv(
                csv_path,
                on_bad_lines='skip',  # skipping bad rows/lines
                encoding='utf-8',
                low_memory=False,
            )
        except TypeError:
            df = pd.read_csv(
                csv_path,
                encoding='utf-8',
                low_memory=False,
                error_bad_lines=False,
                warn_bad_lines=False,
            )
        
        # getting schema before potentially dropping rows
        result['schema'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                original_rows = sum(1 for line in f) - 1 
            result['original_rows'] = max(original_rows, len(df))
        except:
            result['original_rows'] = len(df)
        
        df_cleaned = df
        
        result['cleaned_rows'] = len(df_cleaned)
        result['rows_dropped'] = result['original_rows'] - result['cleaned_rows']
        
        clean_path = csv_path.parent / csv_path.name.replace('.csv', '_clean.csv')
        df_cleaned.to_csv(clean_path, index=False)
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def main():
    """Main cleaning script."""
    
    PROJECT_ROOT = find_project_root()
    GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Generated dir: {GENERATED_DIR}")
    print()
    
    print("Discovering raw_finals CSV files ...")
    csv_files = discover_raw_finals_files(GENERATED_DIR)
    print(f"Found {len(csv_files)} raw_finals files to clean")
    print()
    
    print("=" * 80)
    print("CLEANING RAW FINALS FILES")
    print("=" * 80)
    print()
    
    summary = {
        'total_files': len(csv_files),
        'cleaned_files': 0,
        'failed_files': 0,
        'total_original_rows': 0,
        'total_cleaned_rows': 0,
        'total_rows_dropped': 0
    }
    
    for csv_file in csv_files:
        print(f"Cleaning: {csv_file.name}")
        
        result = clean_csv_file(csv_file)
        
        if result['error']:
            print(f"  ERROR: {result['error']}")
            summary['failed_files'] += 1
        else:
            summary['cleaned_files'] += 1
            summary['total_original_rows'] += result['original_rows']
            summary['total_cleaned_rows'] += result['cleaned_rows']
            summary['total_rows_dropped'] += result['rows_dropped']
            
            print(f"  Original: {result['original_rows']} rows")
            print(f"  Cleaned: {result['cleaned_rows']} rows")
            print(f"  Dropped: {result['rows_dropped']} rows")
            print(f"  Columns: {len(result['schema'])}")
            
            clean_path = csv_file.parent / csv_file.name.replace('.csv', '_clean.csv')
            print(f"  Saved to: {clean_path.name}")
    
    print()
    print("=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print()
    print(f"Total files processed: {summary['total_files']}")
    print(f"Successfully cleaned: {summary['cleaned_files']}")
    print(f"Failed: {summary['failed_files']}")
    print()
    
    print()
    print("Cleaned files saved with '_clean.csv' suffix")
    
    return summary


if __name__ == "__main__":
    summary = main()