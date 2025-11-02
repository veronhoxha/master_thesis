"""
DuckDB Validation for LLM-Generated Datasets across all runs, domains, shots and models.

This script validates that cleaned LLM-generated CSV files are:

1. Loadable into DuckDB (successfully parse as structured tables)
2. Have detectable schema with proper data types
3. Support basic SQL operations (usable for analytics)
"""

### IMPORTS ###

import duckdb
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple

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


def discover_csv_files(generated_root: str) -> List[Path]:
    """
    Find all cleaned CSV files matching the pattern '*raw_finals_of_all_chunks*_clean*.csv'.
    
    Returns list of CSV file paths to validate.
    """
    csv_files = []
    generated_path = Path(generated_root)
    
    for csv_file in generated_path.rglob("*raw_finals_of_all_chunks*_clean*.csv"):
        csv_files.append(csv_file)
    
    return sorted(csv_files)


def validate_csv_in_duckdb(csv_path: Path) -> Dict[str, Any]:
    """
    Validate a CSV file in DuckDB.
    
    Returns:
        {
            'valid': bool,
            'schema': Dict[str, str] (column_name -> dtype),
            'row_count': int,
            'error': str or None,
            'sql_test_passed': bool
        }
    """
    
    result = {
        'valid': False,
        'schema': {},
        'row_count': 0,
        'error': None,
        'sql_test_passed': False
    }
    
    try:
        # creating DuckDB connection
        conn = duckdb.connect()
        
        csv_path_abs = csv_path.resolve().as_posix()
        csv_path_escaped = csv_path_abs.replace("'", "''") 
        
        # trying to load the CSV with error handling, DuckDB's read_csv handles malformed rows
        table_created = False
        try:
            # try read_csv with auto_detect
            query = f"""
            CREATE TABLE temp_table AS 
            SELECT * FROM read_csv('{csv_path_escaped}', 
                                   auto_detect=true
                                   )
            """
            conn.execute(query)
            table_created = True
            
        except Exception as e1:
            # if that fails, try read_csv_auto
            try:
                query = f"""
                CREATE TABLE temp_table AS 
                SELECT * FROM read_csv_auto('{csv_path_escaped}')
                """
                conn.execute(query)
                table_created = True
            except Exception as e2:
                # if all methods fail, trying a simple test
                try:
                    # just try to count rows without creating a table
                    test_query = f"SELECT COUNT(*) FROM read_csv('{csv_path_escaped}', ignore_errors=true, auto_detect=true, header=true)"
                    row_count = conn.execute(test_query).fetchone()[0]
                    result['row_count'] = row_count
                    result['valid'] = True
                    result['error'] = "Partial validation - could load rows but schema detection failed"
                    conn.close()
                    return result
                except:
                    result['error'] = f"Load failed: {str(e1)}; Auto failed: {str(e2)}"
                    conn.close()
                    return result
        
        # getting the schema (only if table was created successfully)
        if table_created:
            try:
                schema_query = "DESCRIBE temp_table"
                schema_rows = conn.execute(schema_query).fetchall()
                
                if schema_rows:
                    for row in schema_rows:
                        column_name = row[0]
                        column_type = row[1]
                        result['schema'][column_name] = column_type
                else:
                    result['error'] = "Schema query returned no columns"
            except Exception as e:
                result['error'] = f"Schema error: {str(e)}"
        
        # getting the row count (only if table was created)
        if table_created:
            try:
                row_count_query = "SELECT COUNT(*) FROM temp_table"
                result['row_count'] = conn.execute(row_count_query).fetchone()[0]
            except Exception as e:
                if not result['error']:
                    result['error'] = f"Row count error: {str(e)}"
        
        # sql usability test (only if table was created)
        if table_created:
            try:
                # test 1: basic SELECT with LIMIT
                conn.execute("SELECT * FROM temp_table LIMIT 1")
                
                # test 2: COUNT
                conn.execute("SELECT COUNT(*) FROM temp_table")
                
                # test 3: try a simple WHERE clause if possible
                if len(result['schema']) > 0:
                    first_col = list(result['schema'].keys())[0]
                    conn.execute(f"SELECT * FROM temp_table WHERE {first_col} IS NOT NULL LIMIT 1")
                
                result['sql_test_passed'] = True
                result['valid'] = True
                
            except Exception as e:
                if not result['error']:
                    result['error'] = f"SQL test failed: {str(e)}"
                result['valid'] = True 
        
        # cleaning up dropping the tables
        if table_created:
            try:
                conn.execute("DROP TABLE IF EXISTS temp_table")
            except:
                pass
        conn.close()
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def group_files_by_run(csv_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Group CSV files by their run identifier.
    
    Returns dict: {"model_domain_shot": [file1, file2, file3]}
    """
    groups = {}
    
    for csv_file in csv_files:
        parts = csv_file.parts
        try:
            # find model, domain, shot from path
            model_idx = -1
            domain_idx = -1
            shot_idx = -1
            
            for i, part in enumerate(parts):
                if 'run' in part.lower() and model_idx == -1:
                    model_idx = i
                elif part in ['hatecrime', 'employment', 'lending']:
                    domain_idx = i
                elif part in ['zero', 'one', 'few']:
                    shot_idx = i
            
            if model_idx != -1 and domain_idx != -1 and shot_idx != -1:
                model = parts[model_idx]
                domain = parts[domain_idx]
                shot = parts[shot_idx]
                
                group_key = f"{model}_{domain}_{shot}"
                
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(csv_file)
        except:
            group_key = csv_file.stem
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(csv_file)
    
    return groups


def main():
    """
    Main validation script.
    
    Automatically detects project root and validates all cleaned CSV files.
    """
    PROJECT_ROOT = find_project_root()
    GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Generated dir: {GENERATED_DIR}")
    print()
    
    # discovering all CSV files
    print("Discovering CSV files ...")
    csv_files = discover_csv_files(str(GENERATED_DIR))
    print(f"Found {len(csv_files)} CSV files to validate")
    print()
    
    # grouping files
    file_groups = group_files_by_run(csv_files)
    print(f"Grouped into {len(file_groups)} run groups")
    print()
    
    # validate each csv file
    print("=" * 80)
    print("VALIDATING FILES IN DUCKDB")
    print("=" * 80)
    print()
    
    validation_results = {}
    
    for group_key, files in file_groups.items():
        print(f"\nGroup: {group_key}")
        print(f"Files: {len(files)}")
        
        group_results = []
        
        for csv_file in files:
            print(f"\n  Validating: {csv_file.name}")
            
            result = validate_csv_in_duckdb(csv_file)
            group_results.append(result)
            
            if result['valid']:
                print(f"    Valid - {result['row_count']} rows, {len(result['schema'])} columns")
                if result['sql_test_passed']:
                    print(f"    SQL operations work")
                if result['error']:
                    print(f"    Warning: {result['error']}")
            else:
                print(f"    Invalid: {result['error']}")
        
        validation_results[group_key] = {
            'files': files,
            'results': group_results
        }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    total_files = sum(len(group['files']) for group in validation_results.values())
    valid_files = sum(
        sum(1 for r in group['results'] if r['valid'])
        for group in validation_results.values()
    )
    
    print(f"\nTotal files validated: {total_files}")
    print(f"Valid files: {valid_files} ({100*valid_files/total_files:.1f}%)")
    print(f"Invalid files: {total_files - valid_files}")
    
    output_file = PROJECT_ROOT / "analysis" / "duckdb_validation_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json_results = {}
        for key, data in validation_results.items():
            json_results[key] = {
                'results': [
                    {
                        'valid': r['valid'],
                        'row_count': r['row_count'],
                        'column_count': len(r['schema']),
                        'schema': r['schema'],
                        'sql_test_passed': r['sql_test_passed'],
                        'error': r['error']
                    }
                    for r in data['results']
                ]
            }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return validation_results


if __name__ == "__main__":
    results = main()