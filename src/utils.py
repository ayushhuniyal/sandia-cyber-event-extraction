"""
utils.py

Shared utility functions for the cybersecurity event extraction pipeline.

Covers:
  - UMAP visualization (cluster projection and display)
  - Embedding format conversion (parse_embedding)
  - Unicode punctuation stripping (strip_unicode_punct)
  - SVO triple splitting from event template strings
"""

import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import umap
import torch


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_umap(
    X_norm: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str = None,
    figsize: tuple = (10, 7),
    show_noise: bool = True
) -> np.ndarray:
    """
    Project embeddings to 2D with UMAP and color by cluster assignment.

    Noise points (HDBSCAN label = -1) are rendered in transparent gray.
    Cluster points are colored using tab20 colormap, cycling for large
    cluster counts.

    Args:
        X_norm: L2-normalized embedding matrix (n_articles, dim)
        labels: Cluster label array from HDBSCAN (length n_articles)
        title: Plot title
        save_path: If provided, save figure to this path at 150 DPI
        figsize: Figure dimensions in inches
        show_noise: If False, exclude noise points from the plot

    Returns:
        2D UMAP projection array of shape (n_articles, 2)
    """
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    projection = reducer.fit_transform(X_norm)

    unique_labels = sorted(set(labels))
    cmap = cm.tab20
    color_map = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
    color_map[-1] = (0.7, 0.7, 0.7, 0.2)  # noise = transparent gray

    fig, ax = plt.subplots(figsize=figsize)

    # Plot noise underneath clusters
    if show_noise and -1 in unique_labels:
        mask = labels == -1
        ax.scatter(
            projection[mask, 0], projection[mask, 1],
            c=[color_map[-1]], s=1, alpha=0.3, label="Noise"
        )

    for label in [l for l in unique_labels if l != -1]:
        mask = labels == label
        ax.scatter(
            projection[mask, 0], projection[mask, 1],
            c=[color_map[label]], s=3, alpha=0.7
        )

    n_clusters = len([l for l in unique_labels if l != -1])
    ax.set_title(f"{title}\n({n_clusters} clusters)", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return projection


def display_saved_umaps(image_paths: list, ncols: int = 2, figsize: tuple = (16, 18)) -> None:
    """
    Display pre-saved UMAP images in a grid layout.

    Used to present Alpine-generated outputs without re-running
    compute-heavy UMAP projections.

    Args:
        image_paths: List of (filepath, title) tuples
        ncols: Number of columns in the grid
        figsize: Figure dimensions in inches
    """
    nrows = -(-len(image_paths) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for ax, (path, title) in zip(axes.flatten(), image_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes.flatten()[len(image_paths):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def compare_approaches_umap(
    direct_images: list,
    hybrid_images: list,
    figsize: tuple = (18, 12)
) -> None:
    """
    Side-by-side comparison of Direct vs Hybrid UMAP projections.

    Args:
        direct_images: List of (filepath, title) for Direct Approach
        hybrid_images: List of (filepath, title) for Hybrid Approach
        figsize: Figure dimensions in inches
    """
    fig, axes = plt.subplots(2, max(len(direct_images), len(hybrid_images)), figsize=figsize)

    for row, images in enumerate([direct_images, hybrid_images]):
        for col, (path, title) in enumerate(images):
            img = mpimg.imread(path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=10, fontweight="bold")
            axes[row, col].axis("off")

    row_labels = ["Direct Approach", "Hybrid Approach"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold", labelpad=10)

    plt.suptitle(
        "Direct vs Hybrid: Dependency parsing produces coherent event-level clusters",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ── Embedding Utilities ───────────────────────────────────────────────────────

def parse_embedding(x) -> np.ndarray:
    """
    Safely convert a stored embedding to numpy array regardless of format.

    Handles three formats that arise from different storage paths:
      - torch.Tensor: from in-memory pipeline runs
      - np.ndarray: from direct numpy operations
      - str: from CSV round-trips where tensor.__repr__() was saved
             e.g., 'tensor([[0.123, 0.456, ...]])'

    Args:
        x: Embedding in any supported format

    Returns:
        np.ndarray of shape (embedding_dim,)
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().squeeze()
    if isinstance(x, np.ndarray):
        return x.squeeze()
    if isinstance(x, str):
        cleaned = x.strip().replace("tensor(", "").rstrip(")")
        return np.array(eval(cleaned), dtype=float).squeeze()
    raise TypeError(f"Unknown embedding type: {type(x)}")


# ── Text Utilities ────────────────────────────────────────────────────────────

def strip_unicode_punct(s: str) -> str:
    """
    Remove all Unicode punctuation characters from a string.

    Used in GloVe preprocessing to normalize tokens before vocabulary lookup.
    Unicode category 'P' covers all punctuation classes (Pc, Pd, Pe, Pf, Pi, Po, Ps).
    """
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def split_event_template(df: pd.DataFrame, event_col: str = "event_text") -> pd.DataFrame:
    """
    Explode semicolon-joined event templates into one row per SVO triple.

    The dependency parsing pipeline produces event_text as a concatenated
    string of triples separated by semicolons:
      e.g., 'lockbit attack hospital ; fbi issue alert ; microsoft patch cve'

    This function splits and explodes those strings for per-triple embedding
    (used by SecureBERT and GloVe Hybrid approaches).

    Args:
        df: DataFrame with article_id and event_text columns
        event_col: Column containing semicolon-joined SVO triples

    Returns:
        DataFrame with columns: article_id, triple_id, split_text
    """
    df = df.copy()
    df["chunks"] = df[event_col].str.strip("{}").str.split(r"\s*;\s*", regex=True)

    exploded = (
        df[["article_id", "chunks"]]
        .explode("chunks")
        .rename(columns={"chunks": "split_text"})
        .dropna()
        .assign(triple_id=lambda d: d.groupby(level=0).cumcount())
        .reset_index(drop=True)
    )

    return exploded[["article_id", "triple_id", "split_text"]]


def inspect_cluster(
    result_df: pd.DataFrame,
    cluster_id: int,
    text_col: str = "event_text",
    n: int = 5
) -> pd.DataFrame:
    """
    Display articles assigned to a specific cluster for manual inspection.

    Used to validate event-level coherence — e.g., confirming that
    all articles in a cluster describe the same real-world incident.

    Args:
        result_df: Clustered DataFrame with cluster_id column
        cluster_id: The cluster to inspect
        text_col: Column containing article text or event template
        n: Number of articles to display

    Returns:
        DataFrame subset of articles in the specified cluster
    """
    cluster = result_df[result_df["cluster_id"] == cluster_id].copy()
    print(f"Cluster {cluster_id}: {len(cluster)} articles")

    if text_col in cluster.columns:
        for _, row in cluster.head(n).iterrows():
            print(f"\n  [{row.get('article_id', '')[:8]}...] "
                  f"{str(row[text_col])[:150]}...")

    return cluster

