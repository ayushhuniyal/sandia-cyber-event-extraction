"""
clustering.py

HDBSCAN clustering utilities for cybersecurity event extraction pipeline.

Provides:
  - run_hdbscan(): fit HDBSCAN on article-level embeddings
  - evaluate_hdbscan(): evaluate a single parameter combination
  - grid_search(): parallel grid search over HDBSCAN parameters

HDBSCAN was selected over K-means for three reasons:
  1. Cybersecurity reporting is non-uniform in density — ransomware articles
     vastly outnumber niche CVE reports. K-means assumes equal-density
     spherical clusters.
  2. HDBSCAN does not require a predefined k — cluster count emerges from data.
  3. HDBSCAN flags ambiguous articles as noise (-1) rather than forcing
     assignment, preserving cluster purity.

All embeddings are L2-normalized before clustering so Euclidean distance
reflects directional similarity rather than magnitude differences.
"""

import numpy as np
import pandas as pd
import hdbscan

from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

# ── Default Parameters ────────────────────────────────────────────────────────
# Selected via grid search across both Direct and Hybrid approaches.
# Lower EPS = more conservative merging (fewer cross-density merges).
# min_samples = 1 allows sparse events (2-3 articles) to form clusters.

DEFAULT_MIN_CLUSTER_SIZE = 4
DEFAULT_MIN_SAMPLES = 1
DEFAULT_CLUSTER_SELECTION_EPS = 0.0001


# ── Core Clustering ───────────────────────────────────────────────────────────

def run_hdbscan(
    article_df: pd.DataFrame,
    embedding_col: str = "embedding",
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps: float = DEFAULT_CLUSTER_SELECTION_EPS
) -> tuple:
    """
    Run HDBSCAN on article-level embeddings.

    Args:
        article_df: DataFrame with article_id and embedding column
        embedding_col: Name of the column containing embedding arrays
        min_cluster_size: Minimum articles required to form a cluster
        min_samples: Controls conservativeness — higher = fewer, denser clusters
        eps: Density merging threshold (lower = more conservative)

    Returns:
        result_df: article_df with cluster_id and membership_prob columns appended
        X_norm: L2-normalized embedding matrix used for clustering
    """
    X = np.vstack(article_df[embedding_col].values)
    X_norm = normalize(X, norm="l2")  # Euclidean on L2-normalized ≈ cosine distance

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=eps,
        metric="euclidean",
        gen_min_span_tree=True
    )
    clusterer.fit(X_norm)

    result_df = article_df.copy()
    result_df["cluster_id"] = clusterer.labels_
    result_df["membership_prob"] = clusterer.probabilities_

    return result_df, X_norm


def clustering_summary(name: str, result_df: pd.DataFrame, X_norm: np.ndarray) -> dict:
    """
    Compute and print clustering diagnostics for a given model run.

    Metrics reported:
      - n_clusters: Number of non-noise clusters found
      - noise_frac: Fraction of articles labeled as noise (-1)
      - silhouette: Mean silhouette coefficient (higher = better-separated clusters)

    Args:
        name: Display name for this model/run
        result_df: DataFrame with cluster_id column
        X_norm: L2-normalized embedding matrix

    Returns:
        Dict with n_clusters, noise_frac, silhouette
    """
    labels = result_df["cluster_id"].values
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).sum() / len(labels)
    sil = silhouette_score(X_norm, labels) if n_clusters > 1 else None

    print(f"\n{name}")
    print(f"  Clusters found:   {n_clusters}")
    print(f"  Noise fraction:   {noise_frac:.4f}")
    if sil is not None:
        print(f"  Silhouette score: {sil:.4f}")
    else:
        print(f"  Silhouette score: N/A (fewer than 2 clusters)")

    return {"n_clusters": n_clusters, "noise_frac": noise_frac, "silhouette": sil}


# ── Parameter Tuning ──────────────────────────────────────────────────────────

def evaluate_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    eps: float
) -> dict:
    """
    Evaluate a single HDBSCAN parameter combination.

    Designed for parallel execution via joblib. Catches exceptions
    gracefully so a single failed configuration doesn't abort the sweep.

    Args:
        X: L2-normalized embedding matrix
        min_cluster_size, min_samples, eps: HDBSCAN parameters

    Returns:
        Dict with parameter values and resulting metrics
    """
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
            metric="euclidean",
            gen_min_span_tree=True
        ).fit(X)

        labels = clusterer.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = np.sum(labels == -1) / len(labels)
        silhouette = silhouette_score(X, labels) if n_clusters > 1 else None

        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "eps": eps,
            "n_clusters": n_clusters,
            "noise_frac": round(noise_frac, 4),
            "silhouette": round(silhouette, 4) if silhouette else None
        }
    except Exception as e:
        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "eps": eps,
            "n_clusters": None,
            "noise_frac": None,
            "silhouette": None,
            "error": str(e)
        }


def grid_search(
    X_norm: np.ndarray,
    param_grid: dict = None,
    n_jobs: int = -1,
    save_path: str = None
) -> pd.DataFrame:
    """
    Parallel grid search over HDBSCAN parameters.

    Runs on Alpine HPC for full corpus (13,700 articles).
    Uses all available CPU cores by default (n_jobs=-1).

    Args:
        X_norm: L2-normalized embedding matrix
        param_grid: Dict of parameter lists. Defaults to the grid
                    used in the original study.
        n_jobs: Number of parallel workers (-1 = all cores)
        save_path: If provided, save results DataFrame to this CSV path

    Returns:
        DataFrame of parameter combinations and their metrics,
        sorted by n_clusters descending
    """
    if param_grid is None:
        param_grid = {
            "MIN_CLUSTER_SIZE": [2, 4],
            "MIN_SAMPLES": [1, 2, 3, 4],
            "CLUSTER_SELECTION_EPS": [0.0001, 0.001, 0.01, 0.1]
        }

    param_combinations = list(ParameterGrid(param_grid))
    print(f"Running {len(param_combinations)} parameter combinations on {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(evaluate_hdbscan)(
            X_norm,
            p["MIN_CLUSTER_SIZE"],
            p["MIN_SAMPLES"],
            p["CLUSTER_SELECTION_EPS"]
        )
        for p in param_combinations
    )

    results_df = pd.DataFrame(results).sort_values("n_clusters", ascending=False)

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Saved tuning results to {save_path}")

    return results_df

