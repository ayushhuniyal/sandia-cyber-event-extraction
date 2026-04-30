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
