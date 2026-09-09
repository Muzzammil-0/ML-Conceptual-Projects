import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

# ---------- 1. Load — explicit spend columns ----------

import os
from sklearn.datasets import fetch_openml

# Create a local data folder in your current DBSCAN project directory
data_dir = os.path.abspath("./data")
os.makedirs(data_dir, exist_ok=True)

data = fetch_openml(
    'wholesale-customers', 
    version=1, 
    as_frame=True, 
    data_home=data_dir
)
df = data.frame.copy()


y_true = df['Channel'].astype(int).values
spend_cols = ['V2', 'V3', 'V4', 'V5', 'V6', 'V7']
X = df[spend_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 2. k-distance plot ----------
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_dist = np.sort(distances[:, k-1])[::-1]

plt.figure(figsize=(8, 5))
plt.plot(k_dist)
plt.xlabel("Points (sorted)")
plt.ylabel(f"Distance to {k}-th nearest neighbor")
plt.title("k-distance plot — knee gives eps")
plt.grid(alpha=0.3); plt.show()

# ---------- 3. Parameter sweep ----------
print("\n--- DBSCAN sweep ---")
for eps in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    for min_samples in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"  eps={eps:<4} min_samples={min_samples:<3} clusters={n_clusters}  noise={n_noise}")

# ---------- 4. Fit at chosen params ----------
eps_chosen = 2.5
min_samples_chosen = 5

db = DBSCAN(eps=eps_chosen, min_samples=min_samples_chosen).fit(X_scaled)
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"\nDBSCAN eps={eps_chosen}, min_samples={min_samples_chosen}")
print(f"  Clusters: {n_clusters}")
print(f"  Noise:    {n_noise} ({n_noise/len(labels)*100:.1f}%)")

mask = labels != -1
if n_clusters >= 2 and mask.sum() > 1:
    print(f"  Silhouette (excl. noise): {silhouette_score(X_scaled[mask], labels[mask]):.4f}")
print(f"  ARI vs Channel: {adjusted_rand_score(y_true, labels):.4f}")

# ---------- 5. Visualize ----------
pca = PCA(n_components=2).fit(X_scaled)
X_2d = pca.transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for label in sorted(set(labels)):
    m = labels == label
    name = "Noise" if label == -1 else f"Cluster {label}"
    marker = 'x' if label == -1 else 'o'
    axes[0].scatter(X_2d[m, 0], X_2d[m, 1], label=name, alpha=0.6, s=30, marker=marker)
axes[0].set_title(f"DBSCAN (eps={eps_chosen}, min_samples={min_samples_chosen})")
axes[0].legend(); axes[0].grid(alpha=0.3)

for c in np.unique(y_true):
    axes[1].scatter(X_2d[y_true == c, 0], X_2d[y_true == c, 1],
                    label=f"True {c}", alpha=0.6, s=30)
axes[1].set_title("Ground truth (Channel)")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ---------- 6. Cluster profiles ----------
print("\nCluster sizes:")
for label in sorted(set(labels)):
    print(f"  {label}: {(labels == label).sum()}")

df_c = pd.DataFrame(X, columns=spend_cols)
df_c['cluster'] = labels
print("\nMean spend per cluster:")
print(df_c.groupby('cluster').mean().round(0))