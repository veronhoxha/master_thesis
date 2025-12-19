############ BASIC DETAILS ANALYSIS ############

### IMPORTS ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")



###########################



plt.style.use('default')
sns.set_palette("husl")

def find_project_root():
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
ANALYSIS_OUT_DIR = PROJECT_ROOT / "analysis" / "basic_details"
ANALYSIS_OUT_DIR.mkdir(exist_ok=True)


def discover_generation_summaries(base_dir: Path) -> list:
    """Find all generation_summary.json files."""
    summary_files = []
    for model_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        model_run = model_dir.name
        model_base = model_run.split('-run')[0]
        run_num = model_run.split('-run')[1] if '-run' in model_run else None
        
        for domain_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            domain = domain_dir.name
            
            for shot_dir in sorted([p for p in domain_dir.iterdir() if p.is_dir()]):
                shot = shot_dir.name
                csv_dir = shot_dir / "csv"
                
                if csv_dir.is_dir():
                    summary_file = csv_dir / "generation_summary.json"
                    if summary_file.exists():
                        summary_files.append({
                            "model": model_base,
                            "model_run": model_run,
                            "run_num": int(run_num) if run_num else None,
                            "domain": domain,
                            "shot": shot,
                            "summary_path": str(summary_file)
                        })
    
    return summary_files

summary_files = discover_generation_summaries(GENERATED_DIR)

# loading and processing all generation summaries
def load_generation_summary(file_path: str) -> dict:
    """Load a generation summary JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_basic_metrics(summary_data: dict) -> dict:
    """Extract basic metrics from generation summary."""
    if not summary_data:
        return {}
    
    # basic generation info
    metrics = {
        "model": summary_data.get("model", ""),
        "domain": summary_data.get("domain", ""),
        "shot_type": summary_data.get("shot_type", ""),
        "format_type": summary_data.get("format_type", ""),
        "target_rows": summary_data.get("target_rows", 0),
        "achieved_rows": summary_data.get("achieved_rows", 0),
        "duration_s": summary_data.get("duration_s", 0),
        "duration_min": summary_data.get("duration_s", 0) / 60,
        "combined_rows": summary_data.get("combined_rows", 0),
        "run_tag": summary_data.get("run_tag", ""),
        "used_slug": summary_data.get("used_slug", ""),
        "timestamp": summary_data.get("timestamp", "")
    }
    
    # chunk analysis
    chunks = summary_data.get("chunks", [])
    if chunks:
        successful_chunks = [c for c in chunks if c.get("success", False)]
        failed_chunks = [c for c in chunks if not c.get("success", False)]
        
        metrics.update({
            "total_chunks": len(chunks),
            "successful_chunks": len(successful_chunks),
            "failed_chunks": len(failed_chunks),
            "success_rate": len(successful_chunks) / len(chunks) if chunks else 0,
            "avg_chunk_duration": np.mean([c.get("duration_s", 0) for c in chunks]) if chunks else 0,
            "avg_rows_per_chunk": np.mean([c.get("rows_generated", 0) for c in successful_chunks]) if successful_chunks else 0,
            "total_retries": sum(c.get("retries", 0) for c in chunks),
            "avg_retries_per_chunk": np.mean([c.get("retries", 0) for c in chunks]) if chunks else 0
        })
        
        # schema correction analysis
        schema_corrections = [c for c in chunks if c.get("had_schema_mismatch", False)]
        metrics.update({
            "chunks_with_schema_corrections": len(schema_corrections),
            "schema_correction_rate": len(schema_corrections) / len(chunks) if chunks else 0,
            "total_schema_corrections": sum(c.get("schema_corrections_count", 0) for c in chunks),
            "avg_schema_corrections_per_chunk": np.mean([c.get("schema_corrections_count", 0) for c in chunks]) if chunks else 0
        })
        
        error_types = [c.get("error_type") for c in failed_chunks if c.get("error_type")]
        if error_types:
            from collections import Counter
            error_counts = Counter(error_types)
            metrics["most_common_error"] = error_counts.most_common(1)[0][0] if error_counts else None
            metrics["unique_error_types"] = len(error_counts)
        else:
            metrics["most_common_error"] = None
            metrics["unique_error_types"] = 0
    else:
        # default values if no chunks were generated
        metrics.update({
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "success_rate": 0,
            "avg_chunk_duration": 0,
            "avg_rows_per_chunk": 0,
            "total_retries": 0,
            "avg_retries_per_chunk": 0,
            "chunks_with_schema_corrections": 0,
            "schema_correction_rate": 0,
            "total_schema_corrections": 0,
            "avg_schema_corrections_per_chunk": 0,
            "most_common_error": None,
            "unique_error_types": 0
        })
    
    return metrics


# chunk data for time progression analysis
def load_chunk_timing_data():
    """Load chunk level timing data for progression analysis."""
    chunk_data = []
    
    for file_info in summary_files:
        summary_data = load_generation_summary(file_info["summary_path"])
        if not summary_data or 'chunks' not in summary_data:
            continue
            
        model_base = file_info['model']
        domain = file_info['domain']
        shot = file_info['shot']
        run_num = file_info['run_num']
        
        for chunk in summary_data['chunks']:
            if chunk.get('success', False):  # only successful chunks
                chunk_data.append({
                    'model': model_base,
                    'domain': domain,
                    'shot_type': shot,
                    'run_num': run_num,
                    'chunk_id': chunk.get('chunk_id', 0),
                    'duration_s': chunk.get('duration_s', 0),
                    'rows_generated': chunk.get('rows_generated', 0)
                })
    
    return pd.DataFrame(chunk_data)


# load and process all generation summaries
all_metrics = []
for i, file_info in enumerate(summary_files):
    if i % 10 == 0:
        print(f"  {i+1}/{len(summary_files)}...")
    summary_data = load_generation_summary(file_info["summary_path"])
    metrics = extract_basic_metrics(summary_data)
    metrics.update({"model_run": file_info["model_run"], "run_num": file_info["run_num"]})
    all_metrics.append(metrics)

details_df = pd.DataFrame(all_metrics)



def analyze_raw_finals_comprehensive():
    """Analyze all raw_finals files to compare total rows vs pandas readable rows."""
    
    results = []
    for model_dir in sorted([p for p in GENERATED_DIR.iterdir() if p.is_dir()]):
        model_run = model_dir.name
        model_base = model_run.split('-run')[0] if '-run' in model_run else model_run
        run_num = int(model_run.split('-run')[1]) if '-run' in model_run else None
        for domain in ['employment', 'hatecrime', 'lending']:
            domain_dir = model_dir / domain
            if not domain_dir.exists():
                continue
            for shot in ['zero', 'one', 'few']:
                shot_dir = domain_dir / shot
                if not shot_dir.exists():
                    continue
                csv_dir = shot_dir / 'csv'
                if not csv_dir.exists():
                    continue
                # find raw_finals files but exclude _clean versions
                all_raw_finals = list(csv_dir.glob('*raw_finals_of_all_chunks*.csv'))
                raw_finals_files = [f for f in all_raw_finals if '_clean' not in f.name]
                if not raw_finals_files:
                    continue
                for csv_file in raw_finals_files:
                    try:
                        with open(csv_file, 'r', encoding='utf-8') as f:
                            total_lines = sum(1 for _ in f)
                        total_data_rows = max(total_lines - 1, 0)
                        try:
                            df_raw = pd.read_csv(csv_file)
                        except pd.errors.ParserError:
                            df_raw = pd.read_csv(csv_file, on_bad_lines='skip')
                        pandas_rows = len(df_raw)
                        pandas_skipped_rows = max(total_data_rows - pandas_rows, 0)
                        results.append({
                            'model': model_base,
                            'model_run': model_run,
                            'run_num': run_num,
                            'domain': domain,
                            'shot_type': shot,
                            'file_name': csv_file.name,
                            'total_lines_in_file': total_lines,
                            'total_data_rows': total_data_rows,
                            'pandas_readable_rows': pandas_rows,
                            'pandas_skipped_rows': pandas_skipped_rows,
                            'total_rows_lost': pandas_skipped_rows,
                        })
                    except Exception as e:
                        print(f"    Error processing {csv_file.name}: {e}")
                        continue
    return pd.DataFrame(results)


def format_domain_name(domain: str) -> str:
    if domain.lower() == 'hatecrime':
        return 'Hate Crime'
    return domain.title()