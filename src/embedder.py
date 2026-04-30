"""
embedder.py

Embedding utilities for the Direct and Hybrid pipeline approaches.

Covers:
  - Token-based chunking (Direct Approach)
  - E5 instruction-tuned embedding (Direct + Hybrid)
  - SecureBERT 4-layer masked mean pooling (Direct + Hybrid)
  - GloVe word vector averaging (Hybrid only)
  - Article-level aggregation from chunk/triple embeddings

All embedding functions return article-level vectors as numpy arrays.
Normalization is handled downstream in clustering.py.
"""

import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast, RobertaModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Chunking (Direct Approach) ────────────────────────────────────────────────

def chunk_article_by_tokens(text: str, tokenizer, max_len: int = 512, stride: int = 96) -> list:
    """
    Sliding window tokenization for a single article.

    Overlap is applied within articles only — chunks never span article
    boundaries, preventing semantic contamination between documents.

    Args:
        text: Raw article string
        tokenizer: HuggingFace tokenizer
        max_len: Maximum tokens per chunk (512 = BERT architecture limit)
        stride: Token overlap between adjacent chunks (~25% of max_len)

    Returns:
        List of chunk dicts with: chunk_id, input_ids, attention_mask,
        start_token, end_token
    """
    if not text:
        return []

    enc = tokenizer(
        text,
        return_tensors=None,
        truncation=False,       # manual windowing — do not auto-truncate
        padding=False,
        add_special_tokens=False
    )

    input_ids = enc["input_ids"]
    attn = [1] * len(input_ids)  # 1 = real token, 0 = padding

    windows = []
    start, chunk_id = 0, 0

    while start < len(input_ids):
        end = min(start + max_len, len(input_ids))
        windows.append({
            "chunk_id": chunk_id,
            "input_ids": input_ids[start:end],
            "attention_mask": attn[start:end],
            "start_token": start,
            "end_token": end
        })
        if end == len(input_ids):
            break
        start = end - stride
        chunk_id += 1

    return windows


def chunk_corpus(articles: list, tokenizer, max_len: int = 512, stride: int = 96) -> list:
    """
    Apply token-based chunking across all articles.

    Args:
        articles: List of article dicts from preprocessor.df_to_articles()
        tokenizer: HuggingFace tokenizer
        max_len: Token window size
        stride: Overlap in tokens between adjacent windows

    Returns:
        List of chunk dicts with article metadata attached
    """
    all_chunks = []
    for art in articles:
        windows = chunk_article_by_tokens(art["text"], tokenizer, max_len, stride)
        for w in windows:
            w["article_id"] = art["article_id"]
            w["title"] = art.get("title", "")
            w["source"] = art.get("source", "")
            w["date"] = art.get("date", "")
            all_chunks.append(w)
    return all_chunks


# ── Pooling Utilities ─────────────────────────────────────────────────────────

def collate_windows_for_batch(tokenizer, windows: list, pad_to_multiple_of: int = 8) -> dict:
    """Pad a batch of chunk windows to uniform length for batch inference."""
    ids = [torch.tensor(w["input_ids"], dtype=torch.long) for w in windows]
    att = [torch.ones(len(w["input_ids"]), dtype=torch.long) for w in windows]
    return tokenizer.pad(
        {"input_ids": ids, "attention_mask": att},
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=pad_to_multiple_of
    )


@torch.no_grad()
def masked_mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Mean pool over real (non-padded) token positions.

    Used for E5 Direct Approach — E5's instruction-tuned objective
    already produces embeddings suitable for clustering without
    layer averaging.
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    return summed / counts                                          # [B, H]


@torch.no_grad()
def last4_layers_masked_mean(out_hidden_states: tuple, attention_mask: Tensor) -> Tensor:
    """
    Average the final 4 transformer hidden layers before token-level pooling.

    Used for SecureBERT — it lacks an explicit sentence embedding training
    objective, so layer averaging stabilizes its representations for
    downstream clustering tasks.
    """
    layers = out_hidden_states[-4:]
    stacked = torch.stack(layers, dim=0).mean(dim=0)  # [B, T, H]
    return masked_mean_pool(stacked, attention_mask)


# ── Direct Approach Embedding ─────────────────────────────────────────────────

@torch.no_grad()
def embed_chunks(
    all_chunks: list,
    tokenizer,
    model,
    batch_size: int = 64,
    pooling: str = "masked_mean"
) -> pd.DataFrame:
    """
    Embed all chunks in batches. Used in the Direct Approach.

    Args:
        all_chunks: List of chunk dicts from chunk_corpus()
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        batch_size: Chunks per forward pass
        pooling: 'masked_mean' (E5) or 'last4_masked_mean' (SecureBERT)

    Returns:
        DataFrame with one row per chunk and an 'embedding' list column
    """
    use_last4 = (pooling == "last4_masked_mean")
    if use_last4:
        model.config.output_hidden_states = True

    rows, vecs = [], []

    for i in range(0, len(all_chunks), batch_size):
        batch_windows = all_chunks[i:i + batch_size]
        batch = collate_windows_for_batch(tokenizer, batch_windows)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=use_last4
        )

        pooled = (
            last4_layers_masked_mean(out.hidden_states, attention_mask)
            if use_last4
            else masked_mean_pool(out.last_hidden_state, attention_mask)
        )
        pooled = pooled.cpu().to(torch.float32).numpy()

        for j, w in enumerate(batch_windows):
            rows.append({
                "article_id": w["article_id"],
                "chunk_id": w["chunk_id"],
                "start_token": w["start_token"],
                "end_token": w["end_token"],
                "title": w.get("title", ""),
                "source": w.get("source", ""),
                "date": w.get("date", ""),
                "chunk_text": tokenizer.decode(w["input_ids"], skip_special_tokens=True)
            })
        vecs.append(pooled)

    X = np.vstack(vecs).astype("float32")
    df = pd.DataFrame(rows)
    df["embedding"] = [x.tolist() for x in X]
    return df


def aggregate_to_article_level(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse chunk-level embeddings to one vector per article using
    token-weighted averaging.

    Chunks covering more tokens contribute proportionally more to the
    final representation, preventing short final chunks from being
    over-weighted in a simple mean.

    Args:
        chunk_df: DataFrame from embed_chunks() with embedding column

    Returns:
        DataFrame with article_id, embedding, title, source, date
    """
    def weighted_embed(group):
        vecs = np.stack(group["embedding"].to_numpy())
        w = (group["end_token"] - group["start_token"]).to_numpy(dtype=float)
        w = np.clip(w, 1.0, None)
        w = w / w.sum()
        return (vecs * w[:, None]).sum(axis=0)

    article_embeddings = (
        chunk_df.groupby("article_id")
        .apply(weighted_embed)
        .reset_index(name="embedding")
    )

    meta = (
        chunk_df.sort_values("date")
        .groupby("article_id")
        .agg(title=("title", "first"), source=("source", "first"), date=("date", "first"))
        .reset_index()
    )

    return article_embeddings.merge(meta, on="article_id", how="left")


# ── Hybrid Approach: E5 ───────────────────────────────────────────────────────

def average_pool_e5(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Masked average pool for E5 — excludes padding positions."""
    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_texts_e5(
    texts: list,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 512
) -> np.ndarray:
    """
    Embed a list of event template strings using E5 (Hybrid Approach).

    Each text is prefixed with 'query: ' per E5's instruction-tuning
    requirement. Texts exceeding 512 tokens are pre-chunked, each chunk
    embedded independently, then averaged into a single document vector.

    Args:
        texts: List of event template strings (one per article)
        tokenizer: E5 tokenizer
        model: E5 model
        batch_size: Articles per batch
        max_length: Token limit per chunk

    Returns:
        np.ndarray of shape (n_articles, 768)
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with E5"):
        batch_texts = texts[i:i + batch_size]
        batch_embs = []

        for t in batch_texts:
            prefixed = "query: " + t  # required E5 instruction prefix
            tokenized = tokenizer(prefixed, add_special_tokens=False)["input_ids"]

            if len(tokenized) > max_length:
                chunks = [tokenized[j:j + max_length] for j in range(0, len(tokenized), max_length)]
                chunk_texts = [tokenizer.decode(c) for c in chunks]
            else:
                chunk_texts = [prefixed]

            encoded = tokenizer(
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**encoded)

            emb = average_pool_e5(outputs.last_hidden_state, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            doc_emb = emb.mean(dim=0, keepdim=True)
            batch_embs.append(doc_emb)

        batch_embs = torch.cat(batch_embs, dim=0)
        embeddings.append(batch_embs)

    return torch.cat(embeddings, dim=0).cpu().numpy()


# ── Hybrid Approach: SecureBERT ───────────────────────────────────────────────

def securebert_embed_triple(text: str, tokenizer, model) -> np.ndarray:
    """
    Embed a single SVO triple string using SecureBERT.

    Uses 4-layer masked mean pooling — averages final 4 transformer
    hidden layers before pooling over token positions.

    Returns:
        np.ndarray of shape (768,)
    """
    batch = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)

    last4 = out.hidden_states[-4:]
    token_reps = torch.stack(last4, dim=0).mean(dim=0)
    attn = batch["attention_mask"].unsqueeze(-1).to(token_reps.dtype)
    sent_emb = (token_reps * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)

    return sent_emb.squeeze(0).cpu().numpy()


def parse_embedding(x) -> np.ndarray:
    """
    Safely convert a stored embedding to numpy array regardless of format.

    Handles: torch.Tensor, np.ndarray, and string representations
    (e.g., 'tensor([[...]])' from CSV round-trips).
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, str):
        cleaned = x.strip().replace("tensor(", "").rstrip(")")
        return np.array(eval(cleaned), dtype=float)
    raise TypeError(f"Unknown embedding type: {type(x)}")


def aggregate_securebert_hybrid(triples_df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """
    Embed each SVO triple with SecureBERT and average to article level.

    Args:
        triples_df: DataFrame with article_id and split_text columns
        tokenizer: SecureBERT tokenizer
        model: SecureBERT model

    Returns:
        DataFrame with article_id and article_securebert_mean columns
    """
    df = triples_df.copy()
    df["triple_embeddings"] = df["split_text"].apply(
        lambda t: securebert_embed_triple(t, tokenizer, model)
    )
    df["triple_embeddings"] = df["triple_embeddings"].apply(parse_embedding)

    def mean_vec(series):
        arrs = [np.asarray(x, dtype=float) for x in series]
        return np.mean(np.vstack(arrs), axis=0)

    return (
        df.groupby("article_id")["triple_embeddings"]
        .apply(mean_vec)
        .reset_index(name="article_securebert_mean")
    )


# ── Hybrid Approach: GloVe ────────────────────────────────────────────────────

def strip_unicode_punct(s: str) -> str:
    """Remove all Unicode punctuation characters from a string."""
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def embed_article_glove(event_text: str, glove_model) -> np.ndarray:
    """
    Embed a full event template string using GloVe word vectors.

    Pipeline:
        event_text → split by ';' → tokenize each triple → GloVe lookup
        → average word vectors per triple → average triple vectors

    Words not in GloVe vocabulary are excluded.
    Triples with no valid word vectors are excluded.

    Args:
        event_text: Semicolon-joined SVO triple string
        glove_model: Loaded Gensim KeyedVectors model

    Returns:
        np.ndarray of shape (300,) or None if no valid tokens
    """
    triples = [g.strip() for g in str(event_text).split(";") if g.strip()]
    triple_means = []

    for triple in triples:
        tokens = [
            strip_unicode_punct(tok).casefold()
            for tok in triple.split()
        ]
        valid_vecs = [
            glove_model[tok]
            for tok in tokens
            if tok and tok in glove_model.key_to_index
        ]
        if not valid_vecs:
            continue
        triple_means.append(np.stack(valid_vecs).mean(axis=0))

    if not triple_means:
        return None

    return np.stack(triple_means).mean(axis=0)


def aggregate_glove_hybrid(templates_df: pd.DataFrame, glove_model) -> pd.DataFrame:
    """
    Apply GloVe embedding to all articles and return article-level DataFrame.

    Articles with no GloVe-coverable tokens are excluded from output.

    Args:
        templates_df: DataFrame with article_id and event_text columns
        glove_model: Loaded Gensim KeyedVectors model

    Returns:
        DataFrame with article_id and article_glove_mean columns
    """
    records = []
    for _, row in tqdm(templates_df.iterrows(), total=len(templates_df), desc="GloVe embedding"):
        vec = embed_article_glove(row["event_text"], glove_model)
        if vec is not None:
            records.append({"article_id": row["article_id"], "article_glove_mean": vec})

    result = pd.DataFrame(records)
    print(f"GloVe: {len(result):,} articles embedded ({len(templates_df) - len(result):,} excluded — no vocab coverage)")
    return result
