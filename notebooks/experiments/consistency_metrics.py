### IMPORTS ###
from __future__ import annotations
import itertools
import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.stats import ks_2samp

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


###########################


def pairwise_indices(n: int) -> Iterable[Tuple[int, int]]:
    '''Generate all unique pairs of indices from 0 to n-1.'''
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def safe_mean_std(series: pd.Series) -> Tuple[float, float]:
    '''Compute mean and stddev of a numeric series.'''
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=1) if len(s) > 1 else 0.0)


def pooled_std(series_list: Sequence[pd.Series]) -> float:
    '''Compute pooled standard deviation across multiple series.'''
    values = []
    for s in series_list:
        values.append(pd.to_numeric(s, errors="coerce").dropna().values)
    concatenated = np.concatenate([v for v in values if v.size > 0], axis=0) if values else np.array([])
    if concatenated.size <= 1:
        return 0.0
    return float(np.std(concatenated, ddof=1))


def detect_column_types(
    dataframes: Sequence[pd.DataFrame],
    explicit_numeric: Optional[Sequence[str]] = None,
    explicit_categorical: Optional[Sequence[str]] = None,
    explicit_text: Optional[Sequence[str]] = None,
    categorical_unique_threshold: int = 50,
    categorical_fraction_threshold: float = 0.5,
) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    
    '''Detect column types across multiple dataframes.'''
    
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns &= set(df.columns)
    if not common_columns:
        return [], [], [], {}

    numeric_cols = set(explicit_numeric or [])
    categorical_cols = set(explicit_categorical or [])
    text_cols = set(explicit_text or [])
    assignments: Dict[str, str] = {}

    for col in common_columns:
        if col in numeric_cols or col in categorical_cols or col in text_cols:
            continue

        is_numeric_in_all = all(pd.api.types.is_numeric_dtype(df[col]) for df in dataframes)
        if is_numeric_in_all:
            numeric_cols.add(col)
            assignments[col] = "numeric"
            continue

        is_object_like = any(pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) for df in dataframes)
        if not is_object_like:
            text_cols.add(col)
            continue

        unique_counts = []
        lengths = []
        for df in dataframes:
            s = df[col].astype(str).dropna()
            unique_counts.append(s.nunique(dropna=True))
            lengths.append(len(s))
        max_unique = max(unique_counts) if unique_counts else 0
        max_len = max(lengths) if lengths else 0
        unique_fraction = (max_unique / max_len) if max_len > 0 else 0.0

        if (max_unique <= categorical_unique_threshold) or (unique_fraction <= categorical_fraction_threshold):
            categorical_cols.add(col)
            assignments[col] = "categorical"
        else:
            text_cols.add(col)
            assignments[col] = "text"

    numeric = sorted(numeric_cols)
    categorical = sorted(categorical_cols - set(numeric))
    text = sorted(text_cols - set(numeric) - set(categorical))
    return numeric, categorical, text, assignments


def compute_type_alignment(
    assignments: Dict[str, str],
    domain: Optional[str],
    ground_truth: Optional[Dict[str, Dict[str, List[str]]]],
) -> Dict[str, float]:
    
    '''Compute type alignment scores based on ground truth.'''
    
    default_scores = {"numeric": 1.0, "categorical": 1.0, "semantic": 1.0}
    if not ground_truth or not domain or domain not in ground_truth:
        return default_scores

    gt = ground_truth[domain]
    section_map = {
        "numeric": "numeric",
        "categorical": "categorical",
        "semantic": "text",
    }

    for section, gt_key in section_map.items():
        expected_cols = gt.get(gt_key) or []
        if not expected_cols:
            default_scores[section] = 1.0
            continue

        target_type = "text" if section == "semantic" else section
        matches = sum(1 for col in expected_cols if assignments.get(col) == target_type)
        default_scores[section] = matches / len(expected_cols) if expected_cols else 1.0

    return default_scores


def compute_numeric_stability(
    dataframes: Sequence[pd.DataFrame],
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    
    '''Compute numeric stability metrics across multiple dataframes.'''
    
    rows = []
    for col in numeric_columns:
        series_list = [pd.to_numeric(df[col], errors="coerce").dropna() for df in dataframes]
        pooled = pooled_std([s for s in series_list])
        mean_diffs = []
        std_diffs = []
        ks_values = []
        for i, j in pairwise_indices(len(series_list)):
            mi, si = safe_mean_std(series_list[i])
            mj, sj = safe_mean_std(series_list[j])
            if not math.isnan(mi) and not math.isnan(mj):
                mean_diffs.append(abs(mi - mj))
            if not math.isnan(si) and not math.isnan(sj):
                std_diffs.append(abs(si - sj))

            a = series_list[i].values
            b = series_list[j].values
            if a.size > 0 and b.size > 0:
                ks_stat = ks_2samp(a, b, alternative="two-sided", mode="auto").statistic
                ks_values.append(float(ks_stat))

        avg_mean_diff = float(np.nanmean(mean_diffs)) if mean_diffs else float("nan")
        avg_std_diff = float(np.nanmean(std_diffs)) if std_diffs else float("nan")
        avg_ks = float(np.nanmean(ks_values)) if ks_values else float("nan")

        scale = pooled if pooled and pooled > 0 else 1.0
        mean_diff_norm = avg_mean_diff / scale if not math.isnan(avg_mean_diff) else float("nan")
        std_diff_norm = avg_std_diff / scale if not math.isnan(avg_std_diff) else float("nan")

        mean_stability = 1.0 / (1.0 + mean_diff_norm) if not math.isnan(mean_diff_norm) else float("nan")
        std_stability = 1.0 / (1.0 + std_diff_norm) if not math.isnan(std_diff_norm) else float("nan")
        ks_stability = 1.0 - avg_ks if not math.isnan(avg_ks) else float("nan")

        rows.append(
            {
                "column": col,
                "avg_mean_diff": avg_mean_diff,
                "avg_std_diff": avg_std_diff,
                "avg_ks": avg_ks,
                "pooled_std": pooled,
                "mean_stability": mean_stability,
                "std_stability": std_stability,
                "ks_stability": ks_stability,
            }
        )
    return pd.DataFrame(rows)


def value_counts_aligned(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    
    '''Get value counts and categories from a series.'''
    
    counts = series.astype(str).value_counts(dropna=True)
    categories = list(counts.index)
    values = counts.values.astype(float)
    return values, categories


def align_distributions(
    a_values: np.ndarray,
    a_cats: List[str],
    b_values: np.ndarray,
    b_cats: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    
    '''Align two categorical distributions by their categories.'''
    
    all_cats = sorted(set(a_cats) | set(b_cats))
    a_map = {c: v for c, v in zip(a_cats, a_values)}
    b_map = {c: v for c, v in zip(b_cats, b_values)}
    a_aligned = np.array([a_map.get(c, 0.0) for c in all_cats], dtype=float)
    b_aligned = np.array([b_map.get(c, 0.0) for c in all_cats], dtype=float)
    return a_aligned, b_aligned, all_cats


def safe_entropy_from_counts(counts: np.ndarray) -> float:

    '''Compute entropy from counts.'''
    
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(scipy_entropy(p, base=2))


def compute_categorical_stability(
    dataframes: Sequence[pd.DataFrame],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    
    '''Compute categorical stability metrics across multiple dataframes.'''
    
    rows = []
    for col in categorical_columns:
        jaccards = []
        entropy_diffs = []
        freq_corrs = []
        max_cardinality_observed = 0

        series_list = [df[col].astype(str).dropna() for df in dataframes]
        for s in series_list:
            max_cardinality_observed = max(max_cardinality_observed, s.nunique())

        for i, j in pairwise_indices(len(series_list)):
            a = series_list[i]
            b = series_list[j]
            set_a = set(a.unique())
            set_b = set(b.unique())
            union = set_a | set_b
            inter = set_a & set_b
            jaccard = (len(inter) / len(union)) if len(union) > 0 else 1.0
            jaccards.append(float(jaccard))

            a_vals, a_cats = value_counts_aligned(a)
            b_vals, b_cats = value_counts_aligned(b)
            a_aligned, b_aligned, all_cats = align_distributions(a_vals, a_cats, b_vals, b_cats)

            ent_a = safe_entropy_from_counts(a_aligned)
            ent_b = safe_entropy_from_counts(b_aligned)
            entropy_diffs.append(abs(ent_a - ent_b))

            # frequency correlation (Pearson) over aligned counts
            if a_aligned.size > 1 and b_aligned.size > 1 and np.any(a_aligned) and np.any(b_aligned):
                corr_matrix = np.corrcoef(a_aligned, b_aligned)
                corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else float("nan")
                freq_corrs.append(corr)
            else:
                freq_corrs.append(float("nan"))

        avg_jaccard = float(np.nanmean(jaccards)) if jaccards else float("nan")
        avg_entropy_diff = float(np.nanmean(entropy_diffs)) if entropy_diffs else float("nan")
        avg_freq_corr = float(np.nanmean(freq_corrs)) if freq_corrs else float("nan")

        # normalize entropy difference by the maximum possible entropy
        max_entropy = math.log2(max(max_cardinality_observed, 1)) if max_cardinality_observed > 0 else 1.0
        entropy_diff_norm = (avg_entropy_diff / max_entropy) if (not math.isnan(avg_entropy_diff) and max_entropy > 0) else float("nan")

        jaccard_stability = avg_jaccard
        entropy_stability = 1.0 - entropy_diff_norm if not math.isnan(entropy_diff_norm) else float("nan")
        freq_corr_stability = ((avg_freq_corr + 1.0) / 2.0) if not math.isnan(avg_freq_corr) else float("nan")

        rows.append(
            {
                "column": col,
                "avg_jaccard": avg_jaccard,
                "avg_entropy_diff": avg_entropy_diff,
                "avg_freq_corr": avg_freq_corr,
                "jaccard_stability": jaccard_stability,
                "entropy_stability": entropy_stability,
                "freq_corr_stability": freq_corr_stability,
            }
        )
    return pd.DataFrame(rows)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    '''Compute cosine similarity between two vectors.'''
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine similarity expects 1D vectors")
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def embed_text_documents(
    documents_by_run: List[List[str]],
    method: str = "sentence-transformers",
    model_name: Optional[str] = None,
    random_state: int = 42,
) -> List[np.ndarray]:
    '''
    Return a centroid embedding per run.
    
    For reproducibility and consistency, this function ONLY uses sentence-transformers.
    It will raise an ImportError if sentence-transformers is not available.
    '''
    flat_docs = [doc for docs in documents_by_run for doc in docs]
    if not flat_docs:
        return [np.zeros(1, dtype=float) for _ in documents_by_run]

    if method == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer 
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for semantic stability analysis. "
                "Install it with: pip install sentence-transformers\n"
            ) from e
        
        model = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2")
        
        centroids = []
        for docs in documents_by_run:
            if not docs:
                centroids.append(np.zeros(model.get_sentence_embedding_dimension(), dtype=float))
            else:
                emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=False)
                centroids.append(np.mean(emb, axis=0))
        return centroids
    
    elif method == "tfidf":
        # explicit TF-IDF method if needed
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer 
        except ImportError as e:
            raise ImportError(
                "sklearn is required for TF-IDF embeddings. "
                "Install it with: pip install scikit-learn"
            ) from e
        
        vectorizer = TfidfVectorizer(max_features=8192)
        tfidf_matrix = vectorizer.fit_transform(flat_docs)
        centroids = []
        start = 0
        for docs in documents_by_run:
            n = len(docs)
            if n == 0:
                centroids.append(np.zeros(tfidf_matrix.shape[1], dtype=float))
            else:
                slice_matrix = tfidf_matrix[start : start + n].toarray()
                centroids.append(np.mean(slice_matrix, axis=0))
            start += n
        return centroids
    
    else:
        raise ValueError(
            f"Unknown embedding method: '{method}'. "
            "Supported methods: 'sentence-transformers', 'tfidf'"
        )


def compute_semantic_stability(
    dataframes: Sequence[pd.DataFrame],
    text_columns: Sequence[str],
    sample_size: Optional[int] = None,
    random_state: int = 42,
    embedding_method: str = "auto",
    embedding_model_name: Optional[str] = None,
) -> pd.DataFrame:
    '''Compute semantic stability for the provided text columns.'''
    rng = np.random.default_rng(random_state) if sample_size is not None else None
    rows = []
    for col in text_columns:
        documents_by_run: List[List[str]] = []
        for df in dataframes:
            s = df[col].astype(str).dropna()
            if s.empty:
                documents_by_run.append([])
            else:
                if sample_size is None:
                    docs = [str(x) for x in s.tolist()]
                else:
                    n = min(sample_size, len(s))
                    assert rng is not None
                    idx = rng.choice(len(s), size=n, replace=False)
                    docs = [str(x) for x in s.iloc[idx].tolist()]
                documents_by_run.append(docs)

        centroids = embed_text_documents(
            documents_by_run,
            method=embedding_method,
            model_name=embedding_model_name,
            random_state=random_state,
        )

        cosine_sims = []
        for i, j in pairwise_indices(len(centroids)):
            cosine_sims.append(cosine_similarity(centroids[i], centroids[j]))
        avg_cosine_sim = float(np.nanmean(cosine_sims)) if cosine_sims else float("nan")
        semantic_stability = ((avg_cosine_sim + 1.0) / 2.0) if not math.isnan(avg_cosine_sim) else float("nan")

        rows.append(
            {
                "column": col,
                "avg_cosine_similarity": avg_cosine_sim,
                "semantic_stability": semantic_stability,
            }
        )
    return pd.DataFrame(rows)


def aggregate_stability_index(
    numeric_df: Optional[pd.DataFrame],
    categorical_df: Optional[pd.DataFrame],
    semantic_df: Optional[pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    domain: Optional[str] = None,
    ground_truth: Optional[Dict[str, Dict[str, List[str]]]] = None,
    type_alignment: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    
    '''Aggregate consistency index from numeric, categorical, and semantic dataframes.'''

    weights = weights or {"numeric": 0.333, "categorical": 0.333, "semantic": 0.334}
    type_alignment = type_alignment or {"numeric": 1.0, "categorical": 1.0, "semantic": 1.0}

    expected_sections = {"numeric", "categorical", "semantic"}

    weights_sum = sum(weights[k] for k in expected_sections)
    weights = {k: weights[k] / weights_sum for k in expected_sections}

    def safe_mean(values: Iterable[float]) -> float:
        arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
        return float(arr.mean()) if arr.size > 0 else float("nan")

    #  numeric ------
    numeric_score = float("nan")
    if numeric_df is not None and not numeric_df.empty:
        vals = []
        for key in ["mean_stability", "std_stability", "ks_stability"]:
            if key in numeric_df:
                vals.extend([float(x) for x in numeric_df[key].tolist() if not np.isnan(x)])
        numeric_score = safe_mean(vals)

    if not np.isnan(numeric_score):
        numeric_score *= type_alignment.get("numeric", 1.0)
    else:
        numeric_score = 0.0 

    #  categorical 
    categorical_score = float("nan")
    if categorical_df is not None and not categorical_df.empty:
        vals = []
        for key in ["jaccard_stability", "entropy_stability", "freq_corr_stability"]:
            if key in categorical_df:
                vals.extend([float(x) for x in categorical_df[key].tolist() if not np.isnan(x)])
        categorical_score = safe_mean(vals)

    if not np.isnan(categorical_score):
        categorical_score *= type_alignment.get("categorical", 1.0)
    else:
        categorical_score = 0.0 

    # semantic 
    semantic_score = float("nan")
    if semantic_df is not None and not semantic_df.empty:
        if "semantic_stability" in semantic_df:
            vals = [float(x) for x in semantic_df["semantic_stability"].tolist() if not np.isnan(x)]
            semantic_score = safe_mean(vals)

    if not np.isnan(semantic_score):
        semantic_score *= type_alignment.get("semantic", 1.0)
    else:
        semantic_score = 0.0  

    #  overall 
    overall = (
        weights["numeric"] * numeric_score +
        weights["categorical"] * categorical_score +
        weights["semantic"] * semantic_score
    )

    return {
        "numeric": numeric_score,
        "categorical": categorical_score,
        "semantic": semantic_score,
        "overall": overall,
    }


def compute_all_stability(
    dataframes: Sequence[pd.DataFrame],
    explicit_numeric: Optional[Sequence[str]] = None,
    explicit_categorical: Optional[Sequence[str]] = None,
    explicit_text: Optional[Sequence[str]] = None,
    semantic_sample_size: Optional[int] = None,
    random_state: int = 42,
    embedding_method: str = "auto",
    embedding_model_name: Optional[str] = None,
    domain: Optional[str] = None,
    ground_truth: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """ Function to compute all metrics and a stability index."""
    numeric_cols, categorical_cols, text_cols, column_assignments = detect_column_types(
        dataframes,
        explicit_numeric=explicit_numeric,
        explicit_categorical=explicit_categorical,
        explicit_text=explicit_text,
    )

    numeric_df = compute_numeric_stability(dataframes, numeric_cols) if numeric_cols else pd.DataFrame()
    categorical_df = compute_categorical_stability(dataframes, categorical_cols) if categorical_cols else pd.DataFrame()
    semantic_df = compute_semantic_stability(
        dataframes,
        text_cols,
        sample_size=semantic_sample_size,
        random_state=random_state,
        embedding_method=embedding_method,
        embedding_model_name=embedding_model_name,
    ) if text_cols else pd.DataFrame()

    type_alignment = compute_type_alignment(column_assignments, domain, ground_truth)

    index_dict = aggregate_stability_index(
        numeric_df if not numeric_df.empty else None,
        categorical_df if not categorical_df.empty else None,
        semantic_df if not semantic_df.empty else None,
        domain=domain,
        ground_truth=ground_truth,
        type_alignment=type_alignment,
    )
    return numeric_df, categorical_df, semantic_df, index_dict