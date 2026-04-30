"""
preprocessor.py

Article formatting pipeline for cybersecurity event extraction.
Converts a cleaned Excel dataset into a list of article dictionaries
with stable MD5-based article IDs for cross-pipeline tracking.

Usage:
    from src.preprocessor import load_articles
    articles = load_articles("data/full_clean.xlsx")

CLI:
    python src/preprocessor.py --input data/full_clean.xlsx
"""

import argparse
import hashlib
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """
    Load and minimally format the cleaned article dataset.

    Expects columns: title, article, source, date.
    Date is coerced to UTC-aware datetime.

    Args:
        path: Path to the cleaned Excel or CSV file

    Returns:
        DataFrame with standardized dtypes
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path, encoding="latin1")
    else:
        df = pd.read_excel(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


def df_to_articles(df: pd.DataFrame, include_title: bool = True) -> list:
    """
    Convert dataframe rows to article dictionaries.

    Each article receives a stable article_id derived from an MD5 hash
    of its text content, enabling consistent cross-pipeline tracking
    without relying on row indices (which can shift after filtering).

    Title is prepended to body text by default — this anchors the
    semantic context of each article during embedding.

    Args:
        df: DataFrame with columns: title, article, source, date
        include_title: If True, prepend title to article body

    Returns:
        List of dicts with keys: article_id, text, source, date, title
    """
    records = []
    for _, row in df.iterrows():
        text = (
            f"{row['title']}\n\n{row['article']}"
            if include_title
            else row["article"]
        )
        # MD5 hash: stable, reproducible, collision-resistant for this scale
        a_id = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        records.append({
            "article_id": a_id,
            "text": text,
            "source": row.get("source", ""),
            "date": row["date"],
            "title": row.get("title", ""),
        })
    return records


def load_articles(path: str, include_title: bool = True) -> list:
    """
    Full pipeline: load raw data and return formatted article list.

    Args:
        path: Path to cleaned dataset (.xlsx or .csv)
        include_title: Whether to prepend title to article body

    Returns:
        List of article dicts ready for chunking or parsing
    """
    df = load_raw(path)
    articles = df_to_articles(df, include_title=include_title)
    print(f"Loaded {len(articles):,} articles from {path}")
    return articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and format article dataset")
    parser.add_argument("--input", required=True, help="Path to cleaned dataset")
    parser.add_argument(
        "--no-title", action="store_true",
        help="Exclude title from article text"
    )
    args = parser.parse_args()

    articles = load_articles(args.input, include_title=not args.no_title)
    print(f"Sample article_id: {articles[0]['article_id']}")
    print(f"Sample text preview: {articles[0]['text'][:200]}")
