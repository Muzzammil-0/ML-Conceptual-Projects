# Hierarchical (Agglomerative) Clustering

Implementing Hierarchical Clustering on the Iris dataset and using a Dendrogram to determine the optimal number of clusters.

## The Concept
Unlike K-Means (which requires you to guess `k` upfront), Hierarchical Clustering builds a tree (Dendrogram) bottom-up. It starts with every point as its own cluster and iteratively merges the two closest clusters until all points belong to one giant cluster.

## Approach
1. Compute the pairwise distances and linkage matrix using `scipy.cluster.hierarchy.linkage` (Ward's method).
2. Plot the Dendrogram to visualize the merges.
3. "Cut" the tree at the largest vertical distance to determine `k`.
4. Fit `AgglomerativeClustering` with the chosen `k`.

## Results
- **Silhouette Score:** 0.5543
- **Cluster Sizes:** [64, 50, 36]
- The 50-cluster perfectly isolated the Setosa species (which is linearly separable), while the algorithm split the overlapping Versicolor/Virginica species into 64 and 36.

## Key Observations
- **No need to guess `k`.** The Dendrogram visually shows you the optimal number of clusters by highlighting the largest distance gaps.
- **Ward's Linkage.** We used Ward's method, which merges clusters by minimizing the variance within the newly formed cluster.
- **Complexity.** Hierarchical clustering is O(N³), making it too slow for large datasets (>10,000 rows). For big data, K-Means or DBSCAN is preferred.