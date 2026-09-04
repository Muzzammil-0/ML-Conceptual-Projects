import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------- 1. Load Data ----------
data = load_iris()
X = data.data

# ---------- 2. Build the Dendrogram ----------
# 'ward' linkage minimizes the variance of the clusters being merged
# It's the most common linkage for Euclidean distance
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
plt.title("Hierarchical Clustering Dendrogram (Iris Dataset)")
plt.xlabel("Sample Index")
plt.ylabel("Distance (Ward)")
# truncate_mode='lastp' shows only the last 20 merged clusters for cleaner visualization
dendrogram(linked, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=12.)
plt.axhline(y=10, color='r', linestyle='--', label='Cut at distance=10 (gives k=3)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---------- 3. Train the Model ----------
# Based on the dendrogram, cutting at distance ~10 gives 3 distinct clusters
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = model.fit_predict(X)

print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
print(f"Cluster sizes: {np.bincount(labels)}")