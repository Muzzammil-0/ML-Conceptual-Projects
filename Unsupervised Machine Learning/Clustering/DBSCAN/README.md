# DBSCAN — Density-Based Clustering on Wholesale Customers

Density-based spatial clustering applied to the UCI Wholesale Customers dataset. Explores parameter tuning via k-distance plots and contrasts the behavioral differences between centroid-based (K-Means) and density-based clustering.

## Dataset
- `fetch_openml('wholesale-customers', version=1)`
- 440 rows, 6 numeric spend features (`V2`–`V7`)
- Standardized prior to clustering (mandatory for density metrics)

## Approach
1. Standardize spend features.
2. Generate a k-distance plot to estimate the optimal `eps` (distance to 5th nearest neighbor).
3. Parameter sweep over `eps` ∈ [1.0, 4.0] and `min_samples` ∈ [3, 10] to observe cluster/noise formation.
4. Fit DBSCAN at the chosen parameters; report cluster count, noise points, and Adjusted Rand Index (ARI).
5. Interpret cluster profiles and compare against the K-Means baseline.

## Results

### Parameter Sweep
- Strict parameters (`eps=1.5, min_samples=3`) yield 3 clusters but with ~5% noise.
- Looser parameters (`eps ≥ 2.0`) collapse the data into a single cluster with ~2–3% noise.

### Final Model (`eps=2.5, min_samples=5`)
- **Clusters:** 1
- **Noise points:** 12 (2.7% of data)
- **ARI vs Channel:** 0.0148

### Cluster Profiles (Mean Spend)
- **Cluster 0 (428 points):** Moderate spend across all categories (V2≈11k, V3≈5k).
- **Noise -1 (12 points):** Extreme, bulk-buying behavior (V2≈34k, V3≈29k).

## Key Observations
- **DBSCAN failed to find the true 2-class split (Horeca vs. Retail).** This is not a bug; it reflects the dataset's geometry. The spending profiles of the two channels are contiguous in 6D space with no clear density gap. K-Means succeeds here because it partitions space by distance to centroids, not by density voids.
- **DBSCAN handles outliers better.** K-Means forced the extreme bulk-buyers into a nonsensical 2-point cluster. DBSCAN correctly flags these 12 extreme spenders as noise/anomalies.
- **Varying density is DBSCAN's weakness.** It assumes a global `eps` threshold. When one segment (retail) is dense and another (bulk buyers) is sparse, no single `eps` can separate them without swallowing everything.

## Stack
scikit-learn · numpy · pandas · matplotlib

## Run
```bash
pip install scikit-learn numpy pandas matplotlib
python wholesale_dbscan.py