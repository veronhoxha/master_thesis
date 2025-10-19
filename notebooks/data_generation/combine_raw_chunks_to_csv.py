"""
Combine raw LLM chunk CSVs (from chunk raws .txt) into a single CSV per run directory.
"""

from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_BASE_DIR = PROJECT_ROOT / "master_thesis" / "data" / "generated"


def dynamic_output_name(csv_dir: Path) -> str:
    try:
        shot = csv_dir.parent.name
        domain = csv_dir.parent.parent.name
        model = csv_dir.parent.parent.parent.name
    except Exception:
        model = csv_dir.parent.parent.parent.name if csv_dir.parent.parent.parent else "model"
        domain = csv_dir.parent.parent.name if csv_dir.parent.parent else "domain"
        shot = csv_dir.parent.name if csv_dir.parent else "shot"
    return f"{model}-raw_finals_of_all_chunks_{domain}_{shot}_csv.csv"


def iter_run_csv_dirs(base_dir: Path) -> List[Path]:
    csv_dirs: List[Path] = []
    if not base_dir.exists():
        return csv_dirs
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for domain_dir in model_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            for shot_dir in domain_dir.iterdir():
                if not shot_dir.is_dir():
                    continue
                csv_dir = shot_dir / "csv"
                if csv_dir.is_dir() and (csv_dir / "generation_log.json").exists():
                    csv_dirs.append(csv_dir)
    return sorted(csv_dirs)


def load_generation_log(csv_dir: Path) -> List[dict]:
    path = csv_dir / "generation_log.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def parsed_rows_for_entry(csv_dir: Path, entry: dict) -> Optional[int]:
    parsed_rel = entry.get("parsed_path")
    if not parsed_rel:
        return None
    f = csv_dir / parsed_rel
    if not f.exists():
        return None
    try:
        cnt = sum(1 for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
        return max(0, cnt - 1)
    except Exception:
        return None


def rows_generated_for_entry(entry: dict) -> Optional[int]:
    try:
        rg = entry.get("rows_generated")
        return int(rg) if rg is not None else None
    except Exception:
        return None


def select_success_raw_paths(csv_dir: Path, log_entries: List[dict]) -> List[Tuple[int, Path, Optional[int]]]:
    selected: List[Tuple[int, Path, Optional[int]]] = []
    for entry in log_entries:
        if not entry.get("success", False):
            continue
        chunk_id = int(entry.get("chunk_id", entry.get("chunk_number", 0)) or 0)
        raw_rel = entry.get("raw_path")
        if not raw_rel:
            att = entry.get("successful_attempt") or entry.get("retries") or 1
            raw_rel = f"chunk_{chunk_id:03d}_attempt_{int(att)}_raw_response.txt"
        raw_path = csv_dir / raw_rel
        if raw_path.exists():
            target_rows = parsed_rows_for_entry(csv_dir, entry)
            if target_rows is None:
                target_rows = rows_generated_for_entry(entry)
            selected.append((chunk_id, raw_path, target_rows))

    pattern_paths = list(csv_dir.glob("chunk_*_raw_response.txt")) + list(csv_dir.glob("chunk_*_attempt_*_raw_response.txt"))
    seen_keys = {(cid, p) for cid, p, _ in [(cid, path, tr) for cid, path, tr in selected]}

    def parse_chunk_attempt(name: str) -> Tuple[Optional[int], int]:
        m = re.search(r"chunk_(\d+)", name)
        cid = int(m.group(1)) if m else None
        m2 = re.search(r"attempt_(\d+)", name)
        att = int(m2.group(1)) if m2 else 1
        return cid, att

    latest_by_chunk: Dict[int, Tuple[int, Path]] = {}
    for p in pattern_paths:
        cid, att = parse_chunk_attempt(p.name)
        if cid is None:
            continue
        if (cid, p) in seen_keys:
            continue
        prev = latest_by_chunk.get(cid)
        if prev is None or att > prev[0]:
            latest_by_chunk[cid] = (att, p)

    for cid, (att, p) in latest_by_chunk.items():
        selected.append((cid, p, None))

    selected.sort(key=lambda x: x[0])
    return selected


def split_lines_keep(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def is_fence(line: str) -> bool:
    t = line.strip()
    return t.startswith("```") or t.startswith("---")


def csv_split_aware(line: str) -> List[str]:
    fields: List[str] = []
    cur = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                cur.append('"')
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue
        if ch == ',' and not in_quotes:
            fields.append(''.join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    fields.append(''.join(cur))
    return fields


def tokenize_header(line: str) -> List[str]:
    return [t.strip().strip('"\'').strip() for t in csv_split_aware(line)]


def clean_header_line(line: str) -> str:
    toks = tokenize_header(line)
    toks = [t for t in toks if t != ""]
    return ",".join(toks)


def looks_like_header(line: str) -> bool:
    toks = tokenize_header(line)
    if len(toks) < 2:
        return False
    alpha_like = sum(1 for t in toks if re.search(r"[A-Za-z]", t or ""))
    digit_like = sum(1 for t in toks if re.fullmatch(r"[-+]?\d+(\.\d+)?", (t or "").strip()) is not None)
    return alpha_like >= max(1, len(toks) // 2) and alpha_like >= digit_like


def header_similarity(a: str, b: str) -> float:
    a_tokens = [t.lower() for t in tokenize_header(a) if t]
    b_tokens = [t.lower() for t in tokenize_header(b) if t]
    if not a_tokens or not b_tokens:
        return 0.0
    set_a, set_b = set(a_tokens), set(b_tokens)
    inter = len(set_a & set_b)
    union = len(set_a | set_b) or 1
    return inter / union


def discover_header_candidates(raw_text: str) -> List[str]:
    cands: List[str] = []
    for ln in split_lines_keep(raw_text):
        if is_fence(ln):
            continue
        if looks_like_header(ln):
            cands.append(clean_header_line(ln))
    return cands


def majority_header(texts: List[str]) -> Optional[str]:
    counter: Counter[str] = Counter()
    for tx in texts:
        for h in discover_header_candidates(tx):
            counter[h] += 1
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def canonical_header_from_existing(csv_dir: Path) -> Optional[str]:
    dyn_out = dynamic_output_name(csv_dir)
    for p in sorted(csv_dir.glob("*_csv.csv")):
        name = p.name
        if name == dyn_out or "-raw_finals_of_all_chunks_" in name:
            continue
        try:
            first = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            if first.strip():
                return clean_header_line(first)
        except Exception:
            continue
    return None


def all_raw_lines_after_canonical_header(raw_text: str, canonical_header_clean: str) -> List[str]:
    lines = split_lines_keep(raw_text)
    start_idx = None
    for i, ln in enumerate(lines):
        if clean_header_line(ln) == canonical_header_clean:
            start_idx = i + 1
            break
    if start_idx is None:
        for i, ln in enumerate(lines):
            if looks_like_header(ln):
                start_idx = i + 1
                break
    if start_idx is None:
        for i, ln in enumerate(lines):
            if ',' in ln:
                start_idx = i
                break
    if start_idx is None:
        return []

    data: List[str] = []
    for ln in lines[start_idx:]:
        if is_fence(ln) or not ln.strip():
            continue
        if clean_header_line(ln) == canonical_header_clean:
            continue
        data.append(ln)
    return data


def remove_header_lines(raw_text: str, canonical_header_clean: str) -> List[str]:
    lines = split_lines_keep(raw_text)
    data: List[str] = []
    for ln in lines:
        if clean_header_line(ln) == canonical_header_clean:
            continue
        data.append(ln)
    return data


def combine_raw_csvs_for_dir(csv_dir: Path) -> Optional[Path]:
    entries = load_generation_log(csv_dir)
    selected = select_success_raw_paths(csv_dir, entries)
    if not selected:
        return None

    texts_by_path: Dict[Path, str] = {}
    for _, raw_path, _ in selected:
        texts_by_path[raw_path] = raw_path.read_text(encoding="utf-8", errors="ignore")

    canonical = canonical_header_from_existing(csv_dir)
    if canonical is None:
        canonical = majority_header(list(texts_by_path.values()))
    if canonical is None:
        first_text = next(iter(texts_by_path.values()))
        for ln in split_lines_keep(first_text):
            if is_fence(ln):
                continue
            if ',' in ln:
                canonical = clean_header_line(ln)
                break
    if canonical is None:
        return None

    out_path = csv_dir / dynamic_output_name(csv_dir)
    with out_path.open("w", encoding="utf-8") as out:
        out.write(canonical + "\n")
        for _, raw_path, target_rows in selected:
            raw_text = texts_by_path[raw_path]
            rows = remove_header_lines(raw_text, canonical)
            for ln in rows:
                out.write(ln.rstrip("\n") + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Combine raw chunk CSVs per run directory")
    parser.add_argument("--model", help="Model dir (exact or base/substring, e.g., llama-3.1-8b-instruct or -run3)")
    parser.add_argument("--domain", help="Domain subdir (e.g., employment, hatecrime, lending)")
    parser.add_argument("--shot", help="Shot subdir (e.g., zero, one, few)")
    parser.add_argument("--base", default=str(DEFAULT_BASE_DIR), help="Base generated dir")
    args = parser.parse_args()

    base_dir = Path(args.base)
    csv_dirs = iter_run_csv_dirs(base_dir)

    def model_matches(model_dirname: str, filter_value: Optional[str]) -> bool:
        if not filter_value:
            return True
        if model_dirname == filter_value:
            return True
        if filter_value in model_dirname:
            return True
        m = re.match(r"^(.*?)-run\d+$", model_dirname)
        if m and (m.group(1) == filter_value):
            return True
        return False

    def matches_filters(csv_dir: Path) -> bool:
        try:
            shot = csv_dir.parent.name
            domain = csv_dir.parent.parent.name
            model = csv_dir.parent.parent.parent.name
        except Exception:
            return False
        if not model_matches(model, args.model):
            return False
        if args.domain and domain != args.domain:
            return False
        if args.shot and shot != args.shot:
            return False
        return True

    filtered_dirs = [d for d in csv_dirs if matches_filters(d)]

    if not filtered_dirs:
        print("No matching csv directories found.")
        print(f"Base searched: {base_dir}")
        print(f"Total csv dirs discovered: {len(csv_dirs)}")
        if csv_dirs[:5]:
            print("Examples:")
            for d in csv_dirs[:5]:
                print(f"  - {d}")
        if args.model or args.domain or args.shot:
            print("try relaxing filters or check exact names.")
        return

    total = 0
    for csv_dir in filtered_dirs:
        out = combine_raw_csvs_for_dir(csv_dir)
        if out:
            total += 1
            print(f"Wrote: {out}")
    print(f"Done. Combined CSVs written for {total} run directories.")


if __name__ == "__main__":
    main()