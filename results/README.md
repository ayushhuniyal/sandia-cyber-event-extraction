# Results

## File Index

### Direct Approach
| File | Description |
|------|-------------|
| `securebert_direct_tuning_results.csv` | HDBSCAN parameter sweep — SecureBERT Direct |
| `securebert_chunks.parquet` | SecureBERT chunk-level embeddings — excluded, exceeds GitHub 25MB limit |

### Hybrid Approach
| File | Description |
|------|-------------|
| `securebert_hybrid_tuning_results.csv` | HDBSCAN parameter sweep — SecureBERT Hybrid |
| `glove_hybrid_tuning_results.csv` | HDBSCAN parameter sweep — GloVe Hybrid |
| `e5_hybrid_embeddings.npy` | E5 article-level embeddings — excluded, exceeds GitHub 25MB limit |
| `e5_hybrid_embeddings.csv` | E5 embeddings with article metadata — excluded, exceeds GitHub 25MB limit |
| `securebert_hybrid_embeddings.csv` | SecureBERT article-level embeddings — excluded, exceeds GitHub 25MB limit |

### Visualizations
| File | Description |
|------|-------------|
| `e5_hybrid_no_noise.png` | Hybrid E5 UMAP — clustered articles only |
| `e5_hybrid_with_noise.png` | Hybrid E5 UMAP — including noise points |
| `securebert_hybrid_no_noise.png` | Hybrid SecureBERT UMAP — clustered only |
| `securebert_hybrid_with_noise.png` | Hybrid SecureBERT UMAP — with noise |
| `glove_hybrid_no_noise.png` | Hybrid GloVe UMAP — clustered only |
| `glove_hybrid_with_noise.png` | Hybrid GloVe UMAP — with noise |

## Clustering Performance Summary

| Model | Clusters | Noise Fraction | Event-Level Coherence |
|-------|----------|----------------|-----------------------|
| Direct E5 | 2 | 0.00 | None |
| Direct SecureBERT | 3 | <0.01 | None |
| Hybrid GloVe | 50–76 | 0.31 | Structural only |
| Hybrid SecureBERT | Moderate | Moderate | Topical |
| Hybrid E5 | 152–269 | Low | Event-level ✓ |

## Notable Clusters (Hybrid E5)

- **Cluster 207** — TikTok service shutdown coverage across multiple outlets
- **Cluster 221** — Palo Alto Networks CVE-2024-3400, including proof-of-concept exploit disclosure
- **Cluster 227** — BlackByte ransomware attack campaign
- **Cluster 185** — Google security update across three CVEs

## Key Parameters

- min_cluster_size: 4
- min_samples: 1
- cluster_selection_epsilon: 0.0001

## Notes on Missing and Excluded Files

**Excluded due to GitHub 25MB file size limit:**
Large embedding matrices are standard practice to exclude from
version control. These files were generated on Google Colab and
the CU Boulder Alpine HPC cluster. To regenerate, run the
embedding cells in `notebooks/01_direct_approach.ipynb` and
`notebooks/02_hybrid_approach.ipynb` with GPU access.

- `securebert_chunks.parquet` — SecureBERT Direct chunk-level embeddings
- `e5_hybrid_embeddings.npy` — E5 Hybrid article-level embeddings
- `e5_hybrid_embeddings.csv` — E5 Hybrid embeddings with metadata
- `securebert_hybrid_embeddings.csv` — SecureBERT Hybrid article-level embeddings

**Not retained after project concluded:**
These files were generated during the project but were not saved
from the HPC cluster or Colab environment before the project ended.
Clustering results and UMAP visualizations derived from these
embeddings are preserved in the notebook outputs and visualizations
above.

- `e5_chunks.parquet` — E5 Direct chunk-level embeddings (Alpine HPC)
- `glove_hybrid_embeddings.csv` — GloVe Hybrid article-level embeddings
- `e5_hybrid_tuning_results.csv` — HDBSCAN parameter sweep, Hybrid E5 pipeline
