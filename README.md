# Unsupervised Cybersecurity Event Extraction
### Identifying and grouping cyber events from 13,700 news articles, no labels required.

## The Problem
Cybersecurity events don't arrive as clean, discrete 
units. The same incident appears across dozens of 
sources as fragments — different outlets covering 
different angles, different timestamps, different 
aspects of the same underlying event. Existing 
detection methods relied on expensive manual 
annotation, which doesn't scale to the speed and 
volume of real-world cyber reporting.

The question: can you identify and group cybersecurity 
events from raw, unannotated news articles?

## The Discovery
Our first approach, embedding full article text and 
clustering directly, failed. Holistic semantic 
embeddings produced a homogeneous manifold: everything 
looked like everything else at the corpus level. The 
Direct E5 pipeline collapsed 13,700 articles into 
two trivial clusters. Zero useful signal.

This failure revealed the actual problem: you cannot 
cluster what you cannot first structure. Full-text 
embeddings capture broad thematic similarity. 
Event detection requires relational structure — 
who did what to whom.

## The Solution
We built a Hybrid pipeline that uses dependency parsing 
as pseudo-annotation — extracting Subject-Verb-Object 
triples from each article before embedding. This 
injected lightweight event structure without any 
labeled data.

Pipeline:
- spaCy dependency parsing → SVO triple extraction
- E5 transformer embeddings on parsed triples
- HDBSCAN* clustering on normalized embeddings
- UMAP visualization for inspection

## Results

| Model | Clusters | Noise Fraction | Coherent Events |
|-------|----------|----------------|-----------------|
| Direct E5 | 2 | 0.00 | None |
| Direct SecureBERT | 3 | <0.01 | None |
| Hybrid GloVe | 50–76 | 0.31 | Structural only |
| Hybrid SecureBERT | Moderate | Moderate | Topical |
| **Hybrid E5** | **152–269** | **Low** | **Event-level** |

Hybrid E5 correctly identified:
- TikTok service shutdown coverage across outlets
- Palo Alto Networks CVE-2024-3400 — including 
  the proof-of-concept exploit disclosure
- BlackByte ransomware attack cluster
- Google security update across three CVEs

## Technical Stack
- Python 3.12, Google Colab + CU Boulder Alpine HPC
- spaCy 3.7.4 (dependency parsing)
- Hugging Face Transformers — E5 (intfloat/e5-base-v2), 
  SecureBERT
- Gensim 4.3.3 (GloVe baseline)
- Scikit-Learn 1.6.1 (HDBSCAN*)
- UMAP (dimensionality reduction)

## Team & Contributions

This project was completed as a research capstone at the University 
of Colorado Boulder in partnership with Sandia National Laboratories, 
in collaboration with three classmates.

**My contributions:**
- Designed and implemented the full preprocessing and chunking 
  pipeline — sliding window tokenization, overlap strategy, and 
  article-level aggregation
- Built the Direct Approach end to end — E5 and SecureBERT embedding 
  pipelines, pooling strategies, and batch inference logic
- Led transformer selection and tradeoff analysis — evaluated E5 vs. 
  SecureBERT vs. GloVe across pooling strategies (masked mean vs. 
  4-layer averaged), identifying why domain adaptation alone was 
  insufficient for event-level separation

## Key Decisions and Tradeoffs

**1. Dependency parsing over full-text embedding**
Holistic article embeddings capture topic similarity, 
not event similarity. Two articles about ransomware 
from different incidents look the same to a full-text 
embedder. SVO triples force the model to compare 
relational structure, who attacked whom, rather 
than broad thematic overlap. This was the load-bearing 
architectural decision.

**2. HDBSCAN over K-means**
Cybersecurity reporting is not uniformly distributed — 
ransomware articles vastly outnumber articles on 
specific CVEs. K-means assumes equal-density spherical 
clusters and requires a predefined k. HDBSCAN finds 
variable-density clusters automatically and flags 
ambiguous articles as noise rather than forcing them 
into a cluster. For a corpus where some events generate 
hundreds of articles and others generate two, 
density-based clustering was the only viable choice.

**3. E5 over SecureBERT as the primary embedding**
SecureBERT's domain-specific training improved 
cybersecurity terminology recognition but overweighted 
lexical similarity — articles using the same jargon 
clustered together regardless of whether they described 
the same event. E5's instruction-tuned training, 
optimized for retrieval and clustering tasks, produced 
more directionally consistent embeddings that separated 
by incident rather than vocabulary. Domain adaptation 
is not always additive.

**4. Article-level averaging of triple embeddings**
Each article produces multiple SVO triples, each 
independently embedded. We averaged these into a 
single article vector. The tradeoff: averaging 
smooths out minor distinctions but substantially 
reduces noise from peripheral sentences. For articles 
describing a central incident, which most 
cybersecurity reports do, this approximation holds.

**5. Treating parsing failures as signal, not noise**
When the dependency parser couldn't identify a 
subject or object, we retained the partial triple 
with a placeholder rather than discarding the 
sentence. This preserved recall at the cost of 
some precision — a deliberate choice given that 
sparse events (reported by only 2-3 outlets) 
would otherwise be lost entirely.

## What To Do Differently
The SVO triple representation is a static snapshot — 
it doesn't capture how events evolve across time 
or how sub-events chain together. An LSTM or 
attention-based sequence model over extracted triples 
could model event progression rather than just 
event grouping. The CVE identifier extraction would also serve 
as a powerful auxiliary signal for cluster filtering 
if integrated into the similarity scoring directly 
rather than used only for post-hoc validation.
