### IMPORTS ###
from __future__ import annotations

import pandas as pd
import os
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

PROJECT_ROOT = find_project_root()
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"

KNOWN_DOMAINS = ("employment", "hatecrime", "lending")
KNOWN_SHOTS = ("zero", "one", "few")

@dataclass
class RunGroup:
    base_model: str
    run_paths: List[str] 


def split_base_and_run(name: str) -> Optional[Tuple[str, str]]:
    lower = name.lower()
    for suffix in ("-run1", "-run2", "-run3"):
        if lower.endswith(suffix):
            return name[: -len(suffix)], name
    return None

def discover_run_groups(generated_root: str = GENERATED_DIR) -> List[RunGroup]:
    entries = [d for d in os.listdir(generated_root) if os.path.isdir(os.path.join(generated_root, d))]
    buckets: Dict[str, List[str]] = {}
    for name in entries:
        split = split_base_and_run(name)
        if not split:
            continue
        base, _ = split
        buckets.setdefault(base, []).append(os.path.join(generated_root, name))

    groups: List[RunGroup] = []
    for base, paths in buckets.items():
        paths_sorted = sorted(paths, key=lambda p: p.lower())
        if len(paths_sorted) >= 2:
            groups.append(RunGroup(base_model=base, run_paths=paths_sorted))
    return groups


def find_present_subdirs(path: str, names: Sequence[str]) -> List[str]:
    present = []
    for n in names:
        p = os.path.join(path, n)
        if os.path.isdir(p):
            present.append(n)
    return present


def load_all_csvs_under(path: str) -> pd.DataFrame:
    """
    Load only CSV files matching the pattern '*-raw_finals_*.csv' from a directory tree,
    Excluding files with '_clean' in the name.
    """
    pattern = os.path.join(path, "**", "*-raw_finals_*.csv")
    all_files = sorted(glob(pattern, recursive=True))
    files = [f for f in all_files if '_clean' not in os.path.basename(f)]
    frames: List[pd.DataFrame] = []
    
    for f in files:
        try:
            try:
                df = pd.read_csv(
                    f,
                    on_bad_lines='skip', 
                    encoding='utf-8',
                    low_memory=False, 
                )
            except TypeError:
                df = pd.read_csv(
                    f,
                    encoding='utf-8',
                    low_memory=False,
                    error_bad_lines=False, 
                    warn_bad_lines=False,
                )
            
            if not df.empty:
                frames.append(df)
                
        except UnicodeDecodeError:
            try:
                try:
                    df = pd.read_csv(
                        f,
                        encoding='latin-1',
                        on_bad_lines='skip',
                        low_memory=False,
                    )
                except TypeError:
                    df = pd.read_csv(
                        f,
                        encoding='latin-1',
                        low_memory=False,
                        error_bad_lines=False,
                        warn_bad_lines=False,
                    )
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"Could not read {f}: {e}")
                continue
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    if not frames:
        return pd.DataFrame()
    
    try:
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)
    except Exception as e:
        print(f"Error concatenating dataframes: {e}")
        return pd.DataFrame()


def load_group_dataframes(
    generated_root: str = GENERATED_DIR,
) -> Dict[Tuple[str, str, str], List[pd.DataFrame]]:

    groups = discover_run_groups(generated_root)
    result: Dict[Tuple[str, str, str], List[pd.DataFrame]] = {}
    for group in groups:
        domains_per_run = [set(find_present_subdirs(p, KNOWN_DOMAINS)) for p in group.run_paths]
        if not domains_per_run:
            continue
        common_domains = set.intersection(*domains_per_run)
        for domain in sorted(common_domains):
            shots_per_run = [set(find_present_subdirs(os.path.join(p, domain), KNOWN_SHOTS)) for p in group.run_paths]
            if not shots_per_run:
                continue
            common_shots = set.intersection(*shots_per_run)
            for shot in sorted(common_shots):
                dfs: List[pd.DataFrame] = []
                for run_path in group.run_paths:
                    leaf = os.path.join(run_path, domain, shot)
                    df = load_all_csvs_under(leaf)
                    dfs.append(df)
                non_empty = sum(1 for d in dfs if not d.empty)
                if non_empty >= 2:
                    result[(group.base_model, domain, shot)] = dfs
    return result