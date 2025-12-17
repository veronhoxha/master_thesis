"""
Analyze Bad Lines in raw CSV Files produced by the LLM's

This script analyzes lines that pandas skips due to CSV parsing errors.
Useful for understanding what LLMs produce that causes bad CSV formatting.

For each raw_finals CSV file, it:

1. Reads line by line
2. Attempts to parse each line
3. Captures lines that cause errors
4. Analyzes patterns (unmatched quotes, empty lines, etc.)
5. Generates statistics and examples

"""

### IMPORTS ###

import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict
import json
from typing import List, Dict, Any, Tuple
import re
import os

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


def extract_metadata_from_path(csv_path: Path, generated_root: Path) -> Dict[str, Any]:
    """
    Extract metadata from file path.
    """
    
    try:
        relative = csv_path.relative_to(generated_root)
        parts = relative.parts
        
        if len(parts) >= 4:
            model_run = parts[0]
            domain = parts[1]
            shot = parts[2]
            
            llm = model_run
            run_num = None
            
            if '-run' in model_run:
                match = re.match(r'^(.*?)-run(\d+)$', model_run)
                if match:
                    llm = match.group(1)
                    run_num = int(match.group(2))
            
            return {
                'llm': llm,
                'model_run': model_run,
                'run': run_num,
                'domain': domain,
                'shot': shot
            }
    except Exception as e:
        pass
    
    return {
        'llm': 'unknown',
        'model_run': 'unknown',
        'run': None,
        'domain': 'unknown',
        'shot': 'unknown'
    }


def discover_raw_finals_files(generated_root: Path):
    """Find all raw_finals CSV files (not cleaned versions)."""
    csv_files = []
    for csv_file in generated_root.rglob("*raw_finals_of_all_chunks*.csv"):
        if "_clean.csv" not in csv_file.name:
            csv_files.append(csv_file)
    return sorted(csv_files)


def detect_csv_error(line: str, expected_cols: int = None, actual_error: str = None, pandas_error_msg: str = None) -> Dict[str, Any]:
    """
    Try to detect what's wrong with a CSV line.
    
    Args:
        line: The CSV line to analyze
        expected_cols: Expected number of columns
        actual_error: Actual error message from pandas if available
    """
    
    result = {
        'error_type': 'unknown',
        'description': '',
        'suspected_cause': '',
        'line_preview': line[:100]
    }
    
    line_stripped = line.rstrip('\n\r')
    
    # checking if line is completely empty
    if not line_stripped.strip():
        result['error_type'] = 'empty_line'
        result['description'] = 'Empty line (only whitespace)'
        result['suspected_cause'] = 'Line contains only whitespace or newlines'
        return result
    
    # checking for unmatched quotes
    quote_count = line.count('"')
    if quote_count % 2 != 0:
        result['error_type'] = 'odd_number_of_symbols'
        result['description'] = f'Odd number of symbols ({quote_count})'
        result['suspected_cause'] = 'There is an odd number of symbols in the line, likely due to a quote that is not closed or opened.'
        return result
    
    if '\n' in line_stripped or '\r' in line_stripped:
        try:
            reader = csv.reader([line])
            list(reader)
        except:
            result['error_type'] = 'row_structure_error'
            result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
            result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
    
    if expected_cols is not None:
        try:
            reader = csv.reader([line])
            parsed = list(reader)
            if parsed and len(parsed[0]) > 0:
                actual_cols = len(parsed[0])
                if actual_cols > expected_cols:
                    result['error_type'] = 'too_many_fields'
                    result['description'] = f'Row has {actual_cols} columns but header expects {expected_cols} columns ({actual_cols - expected_cols} extra)'
                    result['suspected_cause'] = f'Likely due to LLM hallucination, where the model generates more fields than expected.'
                elif actual_cols < expected_cols:
                    result['error_type'] = 'too_few_fields'
                    result['description'] = f'Row has {actual_cols} columns but header expects {expected_cols} columns ({expected_cols - actual_cols} missing)'
                    result['suspected_cause'] = f'Missing {expected_cols - actual_cols} column(s) - likely missing commas between fields, line was truncated, or empty fields at the end were omitted'
                else:
                    pass
            else:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                return result
        except csv.Error as e:
            result['error_type'] = 'row_structure_error'
            result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
            result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
        except Exception:
            pass
    
    # checking for quotes inside unquoted fields
    if '",' in line or line.startswith('"'):
        quoted_sections = re.findall(r'"[^"]*"', line)
        unquoted_parts = re.split(r'"[^"]*"', line)
        
        has_unescaped_quotes = False
        for part in unquoted_parts:
            if part and ('"' in part or part.startswith('"') or part.endswith('"')):
                has_unescaped_quotes = True
                break
        
        if has_unescaped_quotes:
            quote_count = line.count('"')
            if quote_count % 2 != 0:
                result['error_type'] = 'odd_number_of_symbols'
                result['description'] = f'Odd number of symbols ({quote_count})'
                result['suspected_cause'] = 'There is an odd number of symbols in the line, likely due to a quote that is not closed or opened.'
            else:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
    
    try:
        line.encode('utf-8')
    except UnicodeEncodeError:
        result['error_type'] = 'row_structure_error'
        result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
        result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
        return result
    
    consecutive_commas_match = re.search(r',{3,}', line)
    if consecutive_commas_match:
        if expected_cols is not None:
            try:
                reader = csv.reader([line])
                parsed = list(reader)
                if parsed and len(parsed[0]) > 0:
                    actual_cols = len(parsed[0])
                    if actual_cols > expected_cols:
                        result['error_type'] = 'too_many_fields'
                        result['description'] = f'Row has {actual_cols} columns but header expects {expected_cols} columns ({actual_cols - expected_cols} extra)'
                        result['suspected_cause'] = f'Likely due to LLM hallucination, where the model generates more fields than expected.'
                        return result
                    elif actual_cols < expected_cols:
                        result['error_type'] = 'too_few_fields'
                        result['description'] = f'Row has {actual_cols} columns but header expects {expected_cols} columns ({expected_cols - actual_cols} missing)'
                        result['suspected_cause'] = f'Missing {expected_cols - actual_cols} column(s) - likely missing commas between fields, line was truncated, or empty fields at the end were omitted'
                        return result
            except Exception:
                pass
    
    if pandas_error_msg:
        error_lower = pandas_error_msg.lower()
        
        if 'expected' in error_lower and ('fields' in error_lower or 'field' in error_lower or 'columns' in error_lower):
            expected_match = re.search(r'expected\s+(\d+)', error_lower)
            actual_match = re.search(r'saw\s+(\d+)', error_lower) or re.search(r'got\s+(\d+)', error_lower) or re.search(r'(\d+)\s+fields?', error_lower)
            
            if expected_match and actual_match:
                expected = int(expected_match.group(1))
                actual = int(actual_match.group(1))
                if actual > expected:
                    result['error_type'] = 'too_many_fields'
                    result['description'] = f'Row has {actual} columns but header expects {expected} columns ({actual - expected} extra)'
                    result['suspected_cause'] = f'Likely due to LLM hallucination, where the model generates more fields than expected.'
                    return result
                elif actual < expected:
                    result['error_type'] = 'too_few_fields'
                    result['description'] = f'Row has {actual} columns but header expects {expected} columns ({expected - actual} missing)'
                    result['suspected_cause'] = f'Missing {expected - actual} column(s) - likely missing commas between fields, line was truncated, or empty fields at the end were omitted'
                    return result
                else:
                    result['error_type'] = 'row_structure_error'
                    result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                    result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                    return result
            else:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                return result
        elif 'quote' in error_lower:
            if 'unterminated' in error_lower or 'unclosed' in error_lower:
                quote_count = line.count('"')
                result['error_type'] = 'odd_number_of_symbols'
                result['description'] = f'Odd number of symbols ({quote_count})'
                result['suspected_cause'] = 'There is an odd number of symbols in the line, likely due to a quote that is not closed or opened.'
                return result
            elif 'quotechar' in error_lower or 'quote char' in error_lower:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                return result
            else:
                quote_count = line.count('"')
                if quote_count % 2 != 0:
                    result['error_type'] = 'odd_number_of_symbols'
                    result['description'] = f'Odd number of symbols ({quote_count})'
                    result['suspected_cause'] = 'There is an odd number of symbols in the line, likely due to a quote that is not closed or opened.'
                else:
                    result['error_type'] = 'row_structure_error'
                    result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                    result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                return result
        elif 'delimiter' in error_lower:
            result['error_type'] = 'row_structure_error'
            result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
            result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
        elif 'field larger' in error_lower or 'size limit' in error_lower:
            result['error_type'] = 'row_structure_error'
            result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
            result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
    
    if actual_error:
        error_lower = actual_error.lower()
        if 'quote' in error_lower or 'field delimiter' in error_lower:
            # checking if it's an odd number of quotes issue
            quote_count = line.count('"')
            if quote_count % 2 != 0:
                result['error_type'] = 'odd_number_of_symbols'
                result['description'] = f'Odd number of symbols ({quote_count})'
                result['suspected_cause'] = 'There is an odd number of symbols in the line, likely due to a quote that is not closed or opened.'
            else:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
        elif 'column' in error_lower or 'expected' in error_lower:
            result['error_type'] = 'row_structure_error'
            result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
            result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            return result
    
    if result['error_type'] == 'unknown':
        try:
            reader = csv.reader([line])
            parsed = list(reader)
            
            if parsed and len(parsed[0]) > 0:
                actual_cols = len(parsed[0])
                
                if expected_cols is not None:
                    if actual_cols > expected_cols:
                        result['error_type'] = 'too_many_fields'
                        result['description'] = f'Row has {actual_cols} columns but header expects {expected_cols} columns ({actual_cols - expected_cols} extra)'
                        result['suspected_cause'] = f'Likely due to LLM hallucination, where the model generates more fields than expected.'
                        return result
                    elif actual_cols < expected_cols:
                        result['error_type'] = 'too_few_fields'
                        result['description'] = f'Row has {actual_cols} columns but header expects {expected_cols} columns ({expected_cols - actual_cols} missing)'
                        result['suspected_cause'] = f'Missing {expected_cols - actual_cols} column(s) - likely missing commas between fields, line was truncated, or empty fields at the end were omitted'
                        return result
                    else:
                        result['error_type'] = 'row_structure_error'
                        result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                        result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                        return result
                else:
                    result['error_type'] = 'row_structure_error'
                    result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                    result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                    return result
            else:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
                return result
                
        except csv.Error as e:
            error_msg = str(e).lower()
            if 'quote' in error_msg:
                quote_count = line.count('"')
                if quote_count % 2 != 0:
                    result['error_type'] = 'odd_number_of_symbols'
                    result['description'] = f'Odd number of symbols ({quote_count})'
                    result['suspected_cause'] = 'There is an odd number of symbols in the line, likely due to a quote that is not closed or opened.'
                else:
                    result['error_type'] = 'row_structure_error'
                    result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                    result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            elif 'field larger' in error_msg or 'size' in error_msg:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
            else:
                result['error_type'] = 'row_structure_error'
                result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
                result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
        except Exception as e:
            result['error_type'] = 'row_structure_error'
            result['description'] = 'There is some type of issue with the row structure, possibly malformed quoted fields, special characters etc.)'
            result['suspected_cause'] = 'Row structure issue, possibly malformed quoted fields, special characters etc.'
    
    return result


def count_expected_csv_rows(csv_path: Path) -> int:
    """
    Count expected CSV rows.
    """
    logical_rows = 0
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            try:
                for row in reader:
                    logical_rows += 1
            except csv.Error:
                pass
    except Exception:
        pass
    
    if logical_rows > 0:
        return max(logical_rows - 1, 0) 
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1 
    except Exception:
        return 0


def analyze_file_for_bad_lines(csv_path: Path) -> Tuple[List[Dict], int]:
    """
    Analyze a CSV file for bad lines by testing with pandas.
    """
    
    bad_lines = []
    expected_cols = None
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        original_rows = max(total_lines - 1, 0) 
    except Exception:
        original_rows = count_expected_csv_rows(csv_path)
    
    # read with pandas using on_bad_lines='skip' to get actual readable rows
    try:
        df = pd.read_csv(
            csv_path,
            on_bad_lines='skip',
            encoding='utf-8',
            low_memory=False,
        )
        expected_cols = len(df.columns) if not df.empty else None
        cleaned_rows = len(df)
    except TypeError:
        try:
            df = pd.read_csv(
                csv_path,
                encoding='utf-8',
                low_memory=False,
                error_bad_lines=False,
                warn_bad_lines=False,
            )
            expected_cols = len(df.columns) if not df.empty else None
            cleaned_rows = len(df)
        except Exception as e:
            print(f"Error reading file with pandas: {e}")
            return bad_lines, expected_cols
    except Exception as e:
        print(f"Error reading file with pandas: {e}")
        return bad_lines, expected_cols
    
    rows_dropped = max(original_rows - cleaned_rows, 0)
    
    if rows_dropped > 0:
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = list(f)
        except Exception as e:
            print(f"Error reading file line by line: {e}")
            return bad_lines, expected_cols
        
        if len(lines) < 2:
            return bad_lines, expected_cols
        
        header_line = lines[0]
        num_expected_bad = rows_dropped
        
        import tempfile
        
        # tracking which lines we have identified as bad
        bad_line_indices = set()
        
        for i, line in enumerate(lines[1:], start=2):
            line_stripped = line.rstrip('\n\r')
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                    tmp.write(header_line)
                    tmp.write(line)
                    tmp_path = tmp.name
                
                try:
                    test_df = pd.read_csv(tmp_path, on_bad_lines='skip')
                    
                    if len(test_df) == 0:
                        pandas_error_msg = None
                        try:
                            pd.read_csv(tmp_path, on_bad_lines='error', encoding='utf-8', low_memory=False)
                        except pd.errors.ParserError as pe:
                            pandas_error_msg = str(pe)
                        except Exception:
                            pass
                        
                        bad_line_indices.add(i)
                        error_info = detect_csv_error(line, expected_cols, actual_error='Pandas skipped this line', pandas_error_msg=pandas_error_msg)
                        bad_lines.append({
                            'line_number': i,
                            'error_type': error_info['error_type'],
                            'description': error_info['description'],
                            'suspected_cause': error_info['suspected_cause'],
                            'full_line': line_stripped,
                            'line_preview': error_info['line_preview'],
                            'full_line_length': len(line),
                            'actual_error': pandas_error_msg if pandas_error_msg else 'Pandas skipped this line',
                            'has_comma': ',' in line,
                            'has_quote': '"' in line,
                            'has_newline_in_field': '\n' in line or '\r' in line
                        })
                except Exception as parse_err:
                    if i not in bad_line_indices:
                        bad_line_indices.add(i)
                        error_msg = str(parse_err)[:200]
                        pandas_error_msg = error_msg if isinstance(parse_err, pd.errors.ParserError) else None
                        error_info = detect_csv_error(line, expected_cols, actual_error=error_msg, pandas_error_msg=pandas_error_msg)
                        bad_lines.append({
                            'line_number': i,
                            'error_type': error_info['error_type'],
                            'description': error_info['description'],
                            'suspected_cause': error_info['suspected_cause'],
                            'full_line': line_stripped,
                            'line_preview': error_info['line_preview'],
                            'full_line_length': len(line),
                            'actual_error': error_msg,
                            'has_comma': ',' in line,
                            'has_quote': '"' in line,
                            'has_newline_in_field': '\n' in line or '\r' in line
                        })
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            except Exception:
                pass
        
        num_found = len(bad_lines)
        if num_found < num_expected_bad:
            import tempfile

            last_known_good = 1
            last_known_good_rows = 0
            
            for i in range(2, len(lines) + 1):
                if i in bad_line_indices:
                    continue
                
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                        for line_idx in range(i):
                            tmp.write(lines[line_idx])
                    tmp_path = tmp.name
                    
                    try:
                        test_df = pd.read_csv(tmp_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
                        actual_rows = len(test_df)
                        known_bad_in_range = sum(1 for idx in bad_line_indices if 2 <= idx <= i)
                        expected_rows = (i - 1) - known_bad_in_range 
                        
                        if actual_rows < expected_rows:
                            missing_count = expected_rows - actual_rows
                            
                            for j in range(i, last_known_good, -1):
                                if j in bad_line_indices:
                                    continue
                                if len(bad_lines) >= num_expected_bad:
                                    break
                                
                                try:
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp2:
                                        tmp2.write(header_line)
                                        tmp2.write(lines[j-1])
                                    tmp2_path = tmp2.name
                                    
                                    try:
                                        test_df2 = pd.read_csv(tmp2_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
                                        if len(test_df2) == 0:
                                            pandas_error_msg = None
                                            try:
                                                pd.read_csv(tmp2_path, on_bad_lines='error', encoding='utf-8', low_memory=False)
                                            except pd.errors.ParserError as pe:
                                                pandas_error_msg = str(pe)
                                            except Exception:
                                                pass
                                            
                                            bad_line_indices.add(j)
                                            error_info = detect_csv_error(lines[j-1], expected_cols, actual_error='Pandas skipped this line', pandas_error_msg=pandas_error_msg)
                                            bad_lines.append({
                                                'line_number': j,
                                                'error_type': error_info['error_type'],
                                                'description': error_info['description'],
                                                'suspected_cause': error_info['suspected_cause'],
                                                'full_line': lines[j-1].rstrip('\n\r'),
                                                'line_preview': error_info['line_preview'],
                                                'full_line_length': len(lines[j-1]),
                                                'actual_error': pandas_error_msg if pandas_error_msg else 'Pandas skipped this line',
                                                'has_comma': ',' in lines[j-1],
                                                'has_quote': '"' in lines[j-1],
                                                'has_newline_in_field': '\n' in lines[j-1] or '\r' in lines[j-1]
                                            })
                                            
                                            if len(bad_lines) >= num_expected_bad:
                                                break
                                    finally:
                                        try:
                                            os.unlink(tmp2_path)
                                        except:
                                            pass
                                except Exception:
                                    pass
                            
                            remaining_bad = num_expected_bad - len(bad_lines)
                            if remaining_bad > 0:
                                for j in range(i, last_known_good, -1):
                                    if j in bad_line_indices:
                                        continue
                                    if len(bad_lines) >= num_expected_bad:
                                        break
                                    
                                    pandas_error_msg = None
                                    try:
                                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp3:
                                            tmp3.write(header_line)
                                            tmp3.write(lines[j-1])
                                        tmp3_path = tmp3.name
                                        try:
                                            pd.read_csv(tmp3_path, on_bad_lines='error', encoding='utf-8', low_memory=False)
                                        except pd.errors.ParserError as pe:
                                            pandas_error_msg = str(pe)
                                        except Exception:
                                            pass
                                        finally:
                                            try:
                                                os.unlink(tmp3_path)
                                            except:
                                                pass
                                    except Exception:
                                        pass
                                    
                                    bad_line_indices.add(j)
                                    error_msg = f'Line skipped in context (part of multi-line record or context-dependent error)'
                                    error_info = detect_csv_error(lines[j-1], expected_cols, actual_error=error_msg, pandas_error_msg=pandas_error_msg)
                                    bad_lines.append({
                                        'line_number': j,
                                        'error_type': error_info['error_type'],
                                        'description': error_info['description'],
                                        'suspected_cause': error_info['suspected_cause'],
                                        'full_line': lines[j-1].rstrip('\n\r'),
                                        'line_preview': error_info['line_preview'],
                                        'full_line_length': len(lines[j-1]),
                                        'actual_error': pandas_error_msg if pandas_error_msg else error_msg,
                                        'has_comma': ',' in lines[j-1],
                                        'has_quote': '"' in lines[j-1],
                                        'has_newline_in_field': '\n' in lines[j-1] or '\r' in lines[j-1]
                                    })
                                    
                                    if len(bad_lines) >= num_expected_bad:
                                        break
                            
                            if i not in bad_line_indices and len(bad_lines) < num_expected_bad:
                                pandas_error_msg = None
                                try:
                                    pd.read_csv(tmp_path, on_bad_lines='error', encoding='utf-8', low_memory=False)
                                except pd.errors.ParserError as pe:
                                    pandas_error_msg = str(pe)
                                except Exception:
                                    pass
                                
                                bad_line_indices.add(i)
                                error_msg = f'Line skipped in context (expected {expected_rows} rows, got {actual_rows})'
                                error_info = detect_csv_error(lines[i-1], expected_cols, actual_error=error_msg, pandas_error_msg=pandas_error_msg)
                                bad_lines.append({
                                    'line_number': i,
                                    'error_type': error_info['error_type'],
                                    'description': error_info['description'],
                                    'suspected_cause': error_info['suspected_cause'],
                                    'full_line': lines[i-1].rstrip('\n\r'),
                                    'line_preview': error_info['line_preview'],
                                    'full_line_length': len(lines[i-1]),
                                    'actual_error': pandas_error_msg if pandas_error_msg else error_msg,
                                    'has_comma': ',' in lines[i-1],
                                    'has_quote': '"' in lines[i-1],
                                    'has_newline_in_field': '\n' in lines[i-1] or '\r' in lines[i-1]
                                })
                                
                                if len(bad_lines) >= num_expected_bad:
                                    break
                        else:
                            last_known_good = i
                            last_known_good_rows = actual_rows
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                except Exception:
                    pass
                
                if len(bad_lines) >= num_expected_bad:
                    break
        
        num_found = len(bad_lines)
        if num_found != num_expected_bad:
            print(f"Warning: Expected {num_expected_bad} bad lines but found {num_found} for {csv_path.name}")
    
    return bad_lines, expected_cols


def calculate_statistics_by_dimensions(all_bad_lines: List[Dict], file_summaries: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistics broken down by domain, shot, LLM, run, and error type.
    
    Returns:
        Dictionary with various breakdowns and averages
    """
    stats = {}
    
    df_summary = pd.DataFrame(file_summaries)
    
    if all_bad_lines:
        df_bad = pd.DataFrame(all_bad_lines)
        required_cols = ['llm', 'domain', 'shot', 'run']
        for col in required_cols:
            if col not in df_bad.columns:
                df_bad[col] = 'unknown' if col != 'run' else None
    else:
        df_bad = pd.DataFrame()
    
    # statistics by Domain
    if not df_bad.empty:
        stats['by_domain'] = {}
        for domain in df_bad['domain'].unique():
            domain_bad = df_bad[df_bad['domain'] == domain]
            domain_summary = df_summary[df_summary['domain'] == domain]
            
            total_rows_domain = domain_summary['total_rows'].sum()
            bad_rows_domain = len(domain_bad)
            
            stats['by_domain'][domain] = {
                'total_bad_rows': bad_rows_domain,
                'total_rows': int(total_rows_domain),
                'bad_rows_pct': (bad_rows_domain / total_rows_domain * 100) if total_rows_domain > 0 else 0,
                'by_error_type': {}
            }
            
            # by error type for this domain
            for error_type in domain_bad['error_type'].unique():
                error_count = len(domain_bad[domain_bad['error_type'] == error_type])
                stats['by_domain'][domain]['by_error_type'][error_type] = {
                    'count': error_count,
                    'pct_of_total_rows': (error_count / total_rows_domain * 100) if total_rows_domain > 0 else 0,
                    'pct_of_bad_rows': (error_count / bad_rows_domain * 100) if bad_rows_domain > 0 else 0
                }
    
    # statistics by Shot Type
    if not df_bad.empty:
        stats['by_shot'] = {}
        for shot in df_bad['shot'].unique():
            shot_bad = df_bad[df_bad['shot'] == shot]
            shot_summary = df_summary[df_summary['shot'] == shot]
            
            total_rows_shot = shot_summary['total_rows'].sum()
            bad_rows_shot = len(shot_bad)
            
            stats['by_shot'][shot] = {
                'total_bad_rows': bad_rows_shot,
                'total_rows': int(total_rows_shot),
                'bad_rows_pct': (bad_rows_shot / total_rows_shot * 100) if total_rows_shot > 0 else 0,
                'by_error_type': {}
            }
            
            # by error type for this shot
            for error_type in shot_bad['error_type'].unique():
                error_count = len(shot_bad[shot_bad['error_type'] == error_type])
                stats['by_shot'][shot]['by_error_type'][error_type] = {
                    'count': error_count,
                    'pct_of_total_rows': (error_count / total_rows_shot * 100) if total_rows_shot > 0 else 0,
                    'pct_of_bad_rows': (error_count / bad_rows_shot * 100) if bad_rows_shot > 0 else 0
                }
    
    # statistics by LLM
    if not df_bad.empty:
        stats['by_llm'] = {}
        for llm in df_bad['llm'].unique():
            llm_bad = df_bad[df_bad['llm'] == llm]
            llm_summary = df_summary[df_summary['llm'] == llm]
            
            total_rows_llm = llm_summary['total_rows'].sum()
            bad_rows_llm = len(llm_bad)
            
            stats['by_llm'][llm] = {
                'total_bad_rows': bad_rows_llm,
                'total_rows': int(total_rows_llm),
                'bad_rows_pct': (bad_rows_llm / total_rows_llm * 100) if total_rows_llm > 0 else 0,
                'by_error_type': {}
            }
            
            # by error type for this LLM
            for error_type in llm_bad['error_type'].unique():
                error_count = len(llm_bad[llm_bad['error_type'] == error_type])
                stats['by_llm'][llm]['by_error_type'][error_type] = {
                    'count': error_count,
                    'pct_of_total_rows': (error_count / total_rows_llm * 100) if total_rows_llm > 0 else 0,
                    'pct_of_bad_rows': (error_count / bad_rows_llm * 100) if bad_rows_llm > 0 else 0
                }
    
    # statistics by Run
    if not df_bad.empty:
        stats['by_run'] = {}
        for run in sorted([r for r in df_bad['run'].unique() if r is not None]):
            run_bad = df_bad[df_bad['run'] == run]
            run_summary = df_summary[df_summary['run'] == run]
            
            total_rows_run = run_summary['total_rows'].sum()
            bad_rows_run = len(run_bad)
            
            stats['by_run'][f'run{run}'] = {
                'total_bad_rows': bad_rows_run,
                'total_rows': int(total_rows_run),
                'bad_rows_pct': (bad_rows_run / total_rows_run * 100) if total_rows_run > 0 else 0,
                'by_error_type': {}
            }
            
            # by error type for this run
            for error_type in run_bad['error_type'].unique():
                error_count = len(run_bad[run_bad['error_type'] == error_type])
                stats['by_run'][f'run{run}']['by_error_type'][error_type] = {
                    'count': error_count,
                    'pct_of_total_rows': (error_count / total_rows_run * 100) if total_rows_run > 0 else 0,
                    'pct_of_bad_rows': (error_count / bad_rows_run * 100) if bad_rows_run > 0 else 0
                }
    
    # averages across 3 runs (for LLM-Domain-Shot combinations)
    if not df_bad.empty:
        stats['averages_across_runs'] = {}
        
        # grouping by llm, domain, shot
        combinations = df_summary.groupby(['llm', 'domain', 'shot']).agg({
            'total_rows': 'sum',
            'bad_lines_count': 'sum'
        }).reset_index()
        
        for _, row in combinations.iterrows():
            llm = row['llm']
            domain = row['domain']
            shot = row['shot']
            
            combo_bad = df_bad[(df_bad['llm'] == llm) & 
                              (df_bad['domain'] == domain) & 
                              (df_bad['shot'] == shot)]
            combo_summary = df_summary[(df_summary['llm'] == llm) & 
                                      (df_summary['domain'] == domain) & 
                                      (df_summary['shot'] == shot)]
            
            total_rows_combo = combo_summary['total_rows'].sum()
            bad_rows_combo = len(combo_bad)
            num_runs = combo_summary['run'].nunique()
            
            key = f"{llm}_{domain}_{shot}"
            stats['averages_across_runs'][key] = {
                'llm': llm,
                'domain': domain,
                'shot': shot,
                'num_runs': int(num_runs),
                'total_bad_rows': bad_rows_combo,
                'total_rows': int(total_rows_combo),
                'bad_rows_pct': (bad_rows_combo / total_rows_combo * 100) if total_rows_combo > 0 else 0,
                'avg_bad_rows_per_run': bad_rows_combo / num_runs if num_runs > 0 else 0,
                'avg_bad_rows_pct_per_run': (bad_rows_combo / total_rows_combo * 100 / num_runs) if (total_rows_combo > 0 and num_runs > 0) else 0,
                'by_error_type': {}
            }
            
            # by error type for this combination
            for error_type in combo_bad['error_type'].unique():
                error_count = len(combo_bad[combo_bad['error_type'] == error_type])
                stats['averages_across_runs'][key]['by_error_type'][error_type] = {
                    'count': error_count,
                    'pct_of_total_rows': (error_count / total_rows_combo * 100) if total_rows_combo > 0 else 0,
                    'pct_of_bad_rows': (error_count / bad_rows_combo * 100) if bad_rows_combo > 0 else 0,
                    'avg_count_per_run': error_count / num_runs if num_runs > 0 else 0
                }
    
    # by Error Type (overall)
    if not df_bad.empty:
        stats['by_error_type'] = {}
        total_all_rows = df_summary['total_rows'].sum()
        total_all_bad = len(df_bad)
        
        for error_type in df_bad['error_type'].unique():
            error_count = len(df_bad[df_bad['error_type'] == error_type])
            stats['by_error_type'][error_type] = {
                'count': error_count,
                'pct_of_total_rows': (error_count / total_all_rows * 100) if total_all_rows > 0 else 0,
                'pct_of_bad_rows': (error_count / total_all_bad * 100) if total_all_bad > 0 else 0
            }
    
    return stats


def main():
    
    """Main analysis script."""
    
    PROJECT_ROOT = find_project_root()
    GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
    ANALYSIS_OUT = PROJECT_ROOT / "analysis" / "bad_lines"
    ANALYSIS_OUT.mkdir(exist_ok=True)

    print()
    
    print("Discovering raw_finals CSV files ...")
    csv_files = discover_raw_finals_files(GENERATED_DIR)
    print(f"Found {len(csv_files)} raw_finals files to analyze")
    print()
    
    print("=" * 80)
    print("ANALYZING BAD LINES")
    print("=" * 80)
    print()
    
    all_bad_lines = []
    file_summaries = []
    
    for csv_file in csv_files:
        print(f"Analyzing: {csv_file.name}")
        
        bad_lines, expected_cols = analyze_file_for_bad_lines(csv_file)
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                total_lines = len(f.readlines()) - 1 
        except:
            total_lines = len(bad_lines)
        
        metadata = extract_metadata_from_path(csv_file, GENERATED_DIR)
        
        if bad_lines:
            print(f"  Found {len(bad_lines)} bad lines")
            
            error_counts = defaultdict(int)
            for line in bad_lines:
                error_counts[line['error_type']] += 1
            
            print(f"  Error types:")
            for err_type, count in error_counts.items():
                print(f"    {err_type}: {count}")
            
            for line in bad_lines:
                line['file_name'] = csv_file.name
                try:
                    relative_path = csv_file.relative_to(PROJECT_ROOT)
                    line['file_path'] = str(relative_path)
                except:
                    line['file_path'] = csv_file.name
                line.update(metadata) 
                all_bad_lines.append(line)
        else:
            print(f"  No bad lines found - all {total_lines} rows parsed successfully")
        
        try:
            relative_path = csv_file.relative_to(PROJECT_ROOT)
            file_path_str = str(relative_path)
        except:
            file_path_str = csv_file.name
        
        file_summaries.append({
            'file_name': csv_file.name,
            'file_path': file_path_str,
            'total_rows': total_lines,
            'bad_lines_count': len(bad_lines),
            'bad_lines_pct': (len(bad_lines) / total_lines * 100) if total_lines > 0 else 0,
            'expected_columns': expected_cols,
            **metadata 
        })
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if all_bad_lines:
        print(f"Total bad lines found: {len(all_bad_lines)}")
        
        error_types = defaultdict(int)
        for line in all_bad_lines:
            error_types[line['error_type']] += 1
        
        print("\nError types across all files:")
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count} ({count/len(all_bad_lines)*100:.1f}%)")
        
        print("\nExamples of bad lines:")
        print("-" * 80)
        
        examples_shown = set()
        for err_type in sorted(error_types.keys(), key=lambda x: -error_types[x]):
            if err_type not in examples_shown:
                examples_shown.add(err_type)
                # examples of each error type
                for line in all_bad_lines:
                    if line['error_type'] == err_type:
                        print(f"\nError Type: {err_type}")
                        print(f"Cause: {line['suspected_cause']}")
                        print(f"Line number: {line['line_number']} in {line['file_name']}")
                        print(f"Full line: {line['full_line'][:300]}")
                        break
        
        df_bad = pd.DataFrame(all_bad_lines)
        output_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_detailed.csv"
        output_csv.parent.mkdir(exist_ok=True)
        df_bad.to_csv(output_csv, index=False)
        
        df_summary = pd.DataFrame(file_summaries)
        summary_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        
        thesis_data = {
            'total_bad_lines': len(all_bad_lines),
            'error_types': dict(error_types),
            'files_affected': len([f for f in file_summaries if f['bad_lines_count'] > 0]),
            'examples': {}
        }
        
        for err_type in error_types.keys():
            examples_for_type = []
            for line in all_bad_lines:
                if line['error_type'] == err_type:
                    examples_for_type.append({
                        'description': line['description'],
                        'cause': line['suspected_cause'],
                        'full_line': line['full_line'][:500],
                        'file': line['file_name'],
                        'line_number': line['line_number']
                    })
    
                    if len(examples_for_type) >= 3:
                        break
            
            if examples_for_type:
                if len(examples_for_type) == 1:
                    thesis_data['examples'][err_type] = examples_for_type[0]
                else:
                    thesis_data['examples'][err_type] = examples_for_type
        
        thesis_json = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_thesis.json"
        with open(thesis_json, 'w') as f:
            json.dump(thesis_data, f, indent=2)
        
        stats = calculate_statistics_by_dimensions(all_bad_lines, file_summaries)
        
        stats_json = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_statistics.json"
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if 'by_domain' in stats and stats['by_domain']:
            domain_stats_rows = []
            for domain, domain_stats in stats['by_domain'].items():
                for error_type, error_stats in domain_stats['by_error_type'].items():
                    domain_stats_rows.append({
                        'domain': domain,
                        'error_type': error_type,
                        'bad_rows_count': error_stats['count'],
                        'bad_rows_pct_of_total': error_stats['pct_of_total_rows'],
                        'bad_rows_pct_of_bad': error_stats['pct_of_bad_rows'],
                        'total_bad_rows_domain': domain_stats['total_bad_rows'],
                        'total_rows_domain': domain_stats['total_rows'],
                        'bad_rows_pct_domain': domain_stats['bad_rows_pct']
                    })
            
            if domain_stats_rows:
                df_domain = pd.DataFrame(domain_stats_rows)
                domain_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_domain.csv"
                df_domain.to_csv(domain_csv, index=False)
        
        if 'by_shot' in stats and stats['by_shot']:
            shot_stats_rows = []
            for shot, shot_stats in stats['by_shot'].items():
                for error_type, error_stats in shot_stats['by_error_type'].items():
                    shot_stats_rows.append({
                        'shot': shot,
                        'error_type': error_type,
                        'bad_rows_count': error_stats['count'],
                        'bad_rows_pct_of_total': error_stats['pct_of_total_rows'],
                        'bad_rows_pct_of_bad': error_stats['pct_of_bad_rows'],
                        'total_bad_rows_shot': shot_stats['total_bad_rows'],
                        'total_rows_shot': shot_stats['total_rows'],
                        'bad_rows_pct_shot': shot_stats['bad_rows_pct']
                    })
            
            if shot_stats_rows:
                df_shot = pd.DataFrame(shot_stats_rows)
                shot_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_shot.csv"
                df_shot.to_csv(shot_csv, index=False)
        
        if 'by_llm' in stats and stats['by_llm']:
            llm_stats_rows = []
            for llm, llm_stats in stats['by_llm'].items():
                for error_type, error_stats in llm_stats['by_error_type'].items():
                    llm_stats_rows.append({
                        'llm': llm,
                        'error_type': error_type,
                        'bad_rows_count': error_stats['count'],
                        'bad_rows_pct_of_total': error_stats['pct_of_total_rows'],
                        'bad_rows_pct_of_bad': error_stats['pct_of_bad_rows'],
                        'total_bad_rows_llm': llm_stats['total_bad_rows'],
                        'total_rows_llm': llm_stats['total_rows'],
                        'bad_rows_pct_llm': llm_stats['bad_rows_pct']
                    })
            
            if llm_stats_rows:
                df_llm = pd.DataFrame(llm_stats_rows)
                llm_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_llm.csv"
                df_llm.to_csv(llm_csv, index=False)
        
        if 'by_run' in stats and stats['by_run']:
            run_stats_rows = []
            for run, run_stats in stats['by_run'].items():
                for error_type, error_stats in run_stats['by_error_type'].items():
                    run_stats_rows.append({
                        'run': run,
                        'error_type': error_type,
                        'bad_rows_count': error_stats['count'],
                        'bad_rows_pct_of_total': error_stats['pct_of_total_rows'],
                        'bad_rows_pct_of_bad': error_stats['pct_of_bad_rows'],
                        'total_bad_rows_run': run_stats['total_bad_rows'],
                        'total_rows_run': run_stats['total_rows'],
                        'bad_rows_pct_run': run_stats['bad_rows_pct']
                    })
            
            if run_stats_rows:
                df_run = pd.DataFrame(run_stats_rows)
                run_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_run.csv"
                df_run.to_csv(run_csv, index=False)
        
        if 'averages_across_runs' in stats and stats['averages_across_runs']:
            avg_stats_rows = []
            for key, combo_stats in stats['averages_across_runs'].items():
                for error_type, error_stats in combo_stats['by_error_type'].items():
                    avg_stats_rows.append({
                        'llm': combo_stats['llm'],
                        'domain': combo_stats['domain'],
                        'shot': combo_stats['shot'],
                        'num_runs': combo_stats['num_runs'],
                        'error_type': error_type,
                        'total_bad_rows': error_stats['count'],
                        'avg_bad_rows_per_run': error_stats['avg_count_per_run'],
                        'bad_rows_pct_of_total': error_stats['pct_of_total_rows'],
                        'bad_rows_pct_of_bad': error_stats['pct_of_bad_rows'],
                        'total_rows_combo': combo_stats['total_rows'],
                        'total_bad_rows_combo': combo_stats['total_bad_rows'],
                        'bad_rows_pct_combo': combo_stats['bad_rows_pct'],
                        'avg_bad_rows_pct_per_run': combo_stats['avg_bad_rows_pct_per_run']
                    })
            
            if avg_stats_rows:
                df_avg = pd.DataFrame(avg_stats_rows)
                avg_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_averages_across_runs.csv"
                df_avg.to_csv(avg_csv, index=False)
        
        if 'by_error_type' in stats and stats['by_error_type']:
            error_type_rows = []
            for error_type, error_stats in stats['by_error_type'].items():
                error_type_rows.append({
                    'error_type': error_type,
                    'count': error_stats['count'],
                    'pct_of_total_rows': error_stats['pct_of_total_rows'],
                    'pct_of_bad_rows': error_stats['pct_of_bad_rows']
                })
            
            if error_type_rows:
                df_error = pd.DataFrame(error_type_rows)
                error_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_error_type.csv"
                df_error.to_csv(error_csv, index=False)
        
        # group ingbad lines by LLM and Run
        if all_bad_lines:
            df_bad_full = pd.DataFrame(all_bad_lines)
            if not df_bad_full.empty and 'llm' in df_bad_full.columns and 'run' in df_bad_full.columns:
                llm_run_stats_rows = []
                for (llm, run), group in df_bad_full.groupby(['llm', 'run']):
                    if run is None:
                        continue
                    for error_type, error_group in group.groupby('error_type'):
                        llm_run_stats_rows.append({
                            'llm': llm,
                            'run': f'run{run}',
                            'error_type': error_type,
                            'bad_rows_count': len(error_group),
                            'total_bad_rows_llm_run': len(group),
                            'total_bad_rows_llm': len(df_bad_full[df_bad_full['llm'] == llm])
                        })
                
                if llm_run_stats_rows:
                    df_llm_run = pd.DataFrame(llm_run_stats_rows)
                    llm_run_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_llm_run.csv"
                    df_llm_run.to_csv(llm_run_csv, index=False)
        
        # grouping bad lines by LLM, Domain, and Run
        if all_bad_lines:
            df_bad_full = pd.DataFrame(all_bad_lines)
            if not df_bad_full.empty and all(col in df_bad_full.columns for col in ['llm', 'domain', 'run']):
                llm_domain_run_stats_rows = []
                for (llm, domain, run), group in df_bad_full.groupby(['llm', 'domain', 'run']):
                    if run is None:
                        continue
                    for error_type, error_group in group.groupby('error_type'):
                        llm_domain_run_stats_rows.append({
                            'llm': llm,
                            'domain': domain,
                            'run': f'run{run}',
                            'error_type': error_type,
                            'bad_rows_count': len(error_group),
                            'total_bad_rows_combo': len(group)
                        })
                
                if llm_domain_run_stats_rows:
                    df_llm_domain_run = pd.DataFrame(llm_domain_run_stats_rows)
                    llm_domain_run_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_llm_domain_run.csv"
                    df_llm_domain_run.to_csv(llm_domain_run_csv, index=False)
        
        # grouping bad lines by LLM, Domain, Run, and Shot Type
        if all_bad_lines:
            df_bad_full = pd.DataFrame(all_bad_lines)
            if not df_bad_full.empty and all(col in df_bad_full.columns for col in ['llm', 'domain', 'run', 'shot']):
                llm_domain_run_shot_stats_rows = []
                for (llm, domain, run, shot), group in df_bad_full.groupby(['llm', 'domain', 'run', 'shot']):
                    if run is None:
                        continue
                    for error_type, error_group in group.groupby('error_type'):
                        llm_domain_run_shot_stats_rows.append({
                            'llm': llm,
                            'domain': domain,
                            'run': f'run{run}',
                            'shot': shot,
                            'error_type': error_type,
                            'bad_rows_count': len(error_group),
                            'total_bad_rows_combo': len(group)
                        })
                
                if llm_domain_run_shot_stats_rows:
                    df_llm_domain_run_shot = pd.DataFrame(llm_domain_run_shot_stats_rows)
                    llm_domain_run_shot_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_llm_domain_run_shot.csv"
                    df_llm_domain_run_shot.to_csv(llm_domain_run_shot_csv, index=False)
        
        # grouping bad lines by LLM, Domain, and Shot Type (aggregated across all runs)
        if all_bad_lines:
            df_bad_full = pd.DataFrame(all_bad_lines)
            if not df_bad_full.empty and all(col in df_bad_full.columns for col in ['llm', 'domain', 'shot']):
                llm_domain_shot_stats_rows = []
                for (llm, domain, shot), group in df_bad_full.groupby(['llm', 'domain', 'shot']):
                    for error_type, error_group in group.groupby('error_type'):
                        llm_domain_shot_stats_rows.append({
                            'llm': llm,
                            'domain': domain,
                            'shot': shot,
                            'error_type': error_type,
                            'bad_rows_count': len(error_group),
                            'total_bad_rows_combo': len(group)
                        })
                
                if llm_domain_shot_stats_rows:
                    df_llm_domain_shot = pd.DataFrame(llm_domain_shot_stats_rows)
                    llm_domain_shot_csv = PROJECT_ROOT / ANALYSIS_OUT / "bad_lines_by_llm_domain_shot.csv"
                    df_llm_domain_shot.to_csv(llm_domain_shot_csv, index=False)
        
    else:
        print("No bad lines found in any file!")
        print("All CSV files are valid.")
        stats = {}
    
    return {
        'bad_lines': all_bad_lines,
        'summary': file_summaries,
        'total_bad_lines': len(all_bad_lines),
        'statistics': stats
    }


if __name__ == "__main__":
    results = main()