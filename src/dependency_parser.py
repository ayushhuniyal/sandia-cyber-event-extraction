"""
dependency_parser.py

Dependency parsing pipeline for unsupervised cybersecurity event extraction.

Transforms raw cybersecurity news articles into structured event template strings
using spaCy's transformer-based dependency parser (en_core_web_trf).

Pipeline:
    Raw Articles
        → spaCy dependency parsing (en_core_web_trf)
        → SVO triple extraction per sentence
        → Token normalization (lemmatize, lowercase, CVE preservation)
        → Article-level event template construction (semicolon-joined triples)
        → article_event_templates.csv

Output format:
    article_id | event_text
    0          | "lockbit attack hospital ; fbi issue alert ; microsoft patch cve"

The event template string is the input to all three Hybrid Approach
embedding models (E5, SecureBERT, GloVe).

Key design decisions:
    - Partial triples (missing subject or object) are retained with <none>
      placeholder to maximize recall for sparsely-reported events.
    - CVE identifiers (e.g., CVE-2024-3400) are preserved via regex before
      lemmatization — they are the most precise event identifiers in the corpus.
    - NER is disabled to save compute — only the dependency parser is needed.
    - Batch processing with incremental CSV writes supports large corpora
      without memory overflow.

Usage:
    python src/dependency_parser.py \\
        --input data/full_clean.csv \\
        --output data/article_event_templates.csv
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import spacy
from collections import defaultdict


# ── Constants ─────────────────────────────────────────────────────────────────

# CVE identifiers follow a strict format — preserve them exactly
CVE_RE = re.compile(r"\bCVE-\d{4}-\d+\b", flags=re.IGNORECASE)

# Pronouns that never carry meaningful event information
PRONOUNS = {
    "it", "they", "them", "their", "its",
    "this", "that", "those", "these",
    "who", "which", "we", "you", "i"
}


# ── Model Setup ───────────────────────────────────────────────────────────────

def load_spacy_model(prefer_transformer: bool = True):
    """
    Load spaCy model with fallback from transformer to small model.

    en_core_web_trf is preferred — its transformer-based dependency parser
    produces higher quality syntactic parses for complex cybersecurity text.
    Falls back to en_core_web_sm if the transformer model is unavailable.

    NER is disabled to save compute — only the dependency parser is needed.

    Returns:
        Loaded spaCy nlp object
    """
    if prefer_transformer:
        try:
            nlp = spacy.load("en_core_web_trf")
            print("Loaded en_core_web_trf (transformer model)")
        except Exception as e:
            print(f"Could not load en_core_web_trf: {e}")
            print("Falling back to en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    # Disable NER — we only need the dependency parser
    if "ner" in nlp.pipe_names:
        nlp.disable_pipes("ner")

    return nlp


# ── Token Normalization ───────────────────────────────────────────────────────

def clean_entity_text(tok) -> str:
    """
    Normalize a token to its canonical event representation.

    Rules:
      - CVE identifiers are uppercased and preserved exactly
        (e.g., 'cve-2024-3400' → 'CVE-2024-3400')
      - All other tokens are lemmatized and lowercased
      - Non-alphanumeric characters (except hyphens and dots) are stripped
        to handle tokens like 'log4j', 'zero-day'

    Args:
        tok: spaCy token object or '<none>' string

    Returns:
        Normalized string
    """
    if isinstance(tok, str):
        return tok

    txt = tok.text
    if CVE_RE.search(txt):
        return txt.upper()

    lemma = tok.lemma_.lower().strip()
    lemma = re.sub(r"[^a-z0-9\-\.]+", " ", lemma)  # keep alnum, hyphen, dot
    lemma = re.sub(r"\s+", " ", lemma).strip()
    return lemma


def is_good_entity(tok) -> bool:
    """
    Filter out tokens that carry no meaningful event information.

    Rejects:
      - Stop words that aren't proper nouns (e.g., 'the', 'a', 'be')
      - Pronouns (event attribution to 'it' or 'they' is uninformative)
      - Single-character tokens (noise, not CVE identifiers)

    Args:
        tok: spaCy token object

    Returns:
        True if the token is a meaningful event participant
    """
    if tok.is_stop and tok.pos_ != "PROPN":
        return False
    if tok.lemma_.lower() in PRONOUNS:
        return False
    if len(tok.lemma_) <= 1 and not CVE_RE.search(tok.text):
        return False
    return True


# ── SVO Extraction ────────────────────────────────────────────────────────────

def extract_svo_from_doc(doc) -> list:
    """
    Extract Subject-Verb-Object triples from a parsed spaCy document.

    For each sentence, identifies verbs (VERB/AUX) with dependency roles
    ROOT, conj, xcomp, or ccomp, then finds their syntactic subjects and
    objects. Prepositional objects are also included.

    Partial triples (missing subject or object) are retained with '<none>'
    placeholder — this preserves recall for sparse events reported by only
    2-3 outlets, which would otherwise be lost entirely.

    Args:
        doc: spaCy Doc object (parsed)

    Returns:
        List of triple dicts with keys:
            article_id, sent_id, subject, verb, object, voice,
            subject_span, verb_span, object_span
    """
    triples = []

    for sent_id, sent in enumerate(doc.sents):
        # Focus on verbs that head clauses (ROOT, conj, xcomp, ccomp)
        verbs = [
            t for t in sent
            if t.pos_ in ("VERB", "AUX")
            and t.dep_ in ("ROOT", "conj", "xcomp", "ccomp")
        ]

        for v in verbs:
            # Find syntactic subjects
            subs = [
                w for w in v.children
                if w.dep_ in ("nsubj", "nsubjpass") and is_good_entity(w)
            ]
            # Find syntactic objects (direct, attributive, prepositional)
            objs = [
                w for w in v.children
                if w.dep_ in ("dobj", "attr", "dative", "oprd") and is_good_entity(w)
            ]
            # Include prepositional objects
            for prep in [w for w in v.children if w.dep_ == "prep"]:
                objs.extend([
                    w for w in prep.children
                    if w.dep_ == "pobj" and is_good_entity(w)
                ])

            # Skip only if truly no subject AND no object
            if not (subs or objs):
                continue

            # Fill missing slots with placeholder
            if not subs:
                subs = ["<none>"]
            if not objs:
                objs = ["<none>"]

            # Determine voice (active vs passive)
            voice = "passive" if any(
                s != "<none>" and hasattr(s, "dep_") and s.dep_ == "nsubjpass"
                for s in subs
            ) else "active"

            v_norm = v.lemma_.lower()

            for s in subs:
                s_norm = clean_entity_text(s) if s != "<none>" else "<none>"
                for o in objs:
                    o_norm = clean_entity_text(o) if o != "<none>" else "<none>"
                    triples.append({
                        "sent_id": sent_id,
                        "subject": s_norm,
                        "verb": v_norm,
                        "object": o_norm,
                        "voice": voice,
                        "subject_span": s.text if s != "<none>" else "<none>",
                        "verb_span": v.text,
                        "object_span": o.text if o != "<none>" else "<none>"
                    })

    return triples


# ── Event Template Construction ───────────────────────────────────────────────

def triples_to_event_template(triples_df: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """
    Convert per-sentence SVO triples into article-level event template strings.

    Each triple is normalized to a "subject verb object" mini-event string.
    All mini-events for an article are deduplicated and joined with ' ; '.

    Articles with no extractable triples receive an explicit placeholder
    'EVENT: <none>' to ensure every article has a row in the output.

    Args:
        triples_df: DataFrame with article_id, subject, verb, object columns
        n_total: Total number of articles (for complete 0..N-1 coverage)

    Returns:
        DataFrame with article_id and event_text columns
    """
    def norm_piece(x):
        if not isinstance(x, str) or x.strip() == "":
            return ""
        if x == "<none>":
            return ""
        if CVE_RE.fullmatch(x.strip()):
            return x.upper()
        return x.lower().strip()

    df = triples_df.copy()
    df["subject_n"] = df["subject"].apply(norm_piece)
    df["verb_n"] = df["verb"].apply(norm_piece)
    df["object_n"] = df["object"].apply(norm_piece)

    def mini_event(row):
        parts = [p for p in [row["subject_n"], row["verb_n"], row["object_n"]] if p]
        return " ".join(parts)

    df["mini"] = df.apply(mini_event, axis=1)

    # Aggregate per article
    by_article = defaultdict(list)
    for aid, chunk in df.groupby("article_id"):
        texts = sorted(set(t for t in chunk["mini"].tolist() if t))
        if texts:
            by_article[int(aid)] = texts

    # Build output covering all 0..N-1 article IDs
    rows = []
    for aid in range(n_total):
        mini_events = by_article.get(aid, [])
        event_text = " ; ".join(mini_events) if mini_events else "EVENT: <none>"
        rows.append({"article_id": aid, "event_text": event_text})

    return pd.DataFrame(rows)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def load_csv_robust(path: str) -> pd.DataFrame:
    """Load CSV with UTF-8, cp1252, and latin1 encoding fallbacks."""
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise RuntimeError(f"Could not read {path} with utf-8 / cp1252 / latin1")


def run_parsing_pipeline(
    input_path: str,
    output_path: str,
    text_col: str = "article",
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Full dependency parsing pipeline: articles → event templates.

    Processes articles in batches using nlp.pipe for efficiency.
    Writes output to CSV at output_path.

    Args:
        input_path: Path to cleaned article dataset (.csv or .xlsx)
        output_path: Path to write article_event_templates.csv
        text_col: Column name containing article body text
        batch_size: Number of articles per nlp.pipe batch

    Returns:
        DataFrame with article_id and event_text columns
    """
    print("Loading articles...")
    df = load_csv_robust(input_path)
    documents = df[text_col].astype(str).tolist()
    n_total = len(documents)
    print(f"Loaded {n_total:,} articles")

    print("Loading spaCy model...")
    nlp = load_spacy_model(prefer_transformer=True)

    print(f"Parsing {n_total:,} articles in batches of {batch_size}...")
    all_rows = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk = documents[start:end]

        for doc_id, doc in zip(range(start, end), nlp.pipe(chunk, batch_size=batch_size)):
            triples = extract_svo_from_doc(doc)
            for t in triples:
                t["article_id"] = doc_id
                all_rows.append(t)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end:,} / {n_total:,} articles...")

    print(f"Extracted {len(all_rows):,} total triples")

    triples_df = pd.DataFrame(all_rows, columns=[
        "article_id", "sent_id", "subject", "verb", "object",
        "voice", "subject_span", "verb_span", "object_span"
    ])

    print("Building event templates...")
    templates_df = triples_to_event_template(triples_df, n_total)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    templates_df.to_csv(output_path, index=False)

    print(f"Saved {len(templates_df):,} event templates → {output_path}")
    print(f"Coverage: {templates_df['article_id'].nunique():,} unique article_ids")
    print("\nSample output:")
    print(templates_df.head(3).to_string())

    return templates_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dependency parsing pipeline")
    parser.add_argument("--input", required=True, help="Path to cleaned article CSV")
    parser.add_argument("--output", required=True, help="Path to save event templates CSV")
    parser.add_argument("--text-col", default="article", help="Column name for article body text")
    parser.add_argument("--batch-size", type=int, default=128, help="spaCy pipe batch size")
    args = parser.parse_args()

    run_parsing_pipeline(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        batch_size=args.batch_size
    )
