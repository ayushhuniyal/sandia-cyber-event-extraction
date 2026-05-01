# Results

## Clustering Performance Summary

| Model | Clusters | Noise Fraction | Event-Level Coherence |
|-------|----------|----------------|-----------------------|
| Direct E5 | 2 | 0.00 | None |
| Direct SecureBERT | 3 | <0.01 | None |
| Hybrid GloVe | 50–76 | 0.31 | Structural only |
| Hybrid SecureBERT | Moderate | Moderate | Topical |
| Hybrid E5 | 152–269 | Low | Event-level ✓ |

## Notable Clusters (Hybrid E5)
- **Cluster 207** — TikTok service shutdown coverage 
  across multiple outlets
- **Cluster 221** — Palo Alto Networks CVE-2024-3400, 
  including proof-of-concept exploit disclosure
- **Cluster 227** — BlackByte ransomware attack campaign
- **Cluster 185** — Google security update across 
  three CVEs

## Key Parameters
- min_cluster_size: 4
- min_samples: 1  
- cluster_selection_epsilon: 0.0001

## Notes on Missing Files

The following files were generated on the CU Boulder Alpine HPC 
cluster or Google Colab during the project and were not retained 
after the project concluded:

**Direct Approach:**
`e5_chunks.parquet` — E5 Direct chunk-level embeddings. Clustering 
results and UMAP visualizations derived from these embeddings are 
preserved in the notebook outputs.

**Hybrid Approach:**
`glove_hybrid_embeddings.csv` — GloVe article-level embeddings. 
Only the clustered output (dependency_glove_with_cluster_id.csv) 
was retained. The pre-cluster embedding matrix was not saved 
separately.

`e5_hybrid_tuning_results.csv` — HDBSCAN parameter sweep results 
for the Hybrid E5 pipeline. Final selected parameters are 
documented in Key Parameters above.

All other Hybrid Approach files (E5 embeddings .npy/.csv, 
SecureBERT hybrid embeddings, GloVe and SecureBERT tuning results) 
are fully preserved and loadable.
