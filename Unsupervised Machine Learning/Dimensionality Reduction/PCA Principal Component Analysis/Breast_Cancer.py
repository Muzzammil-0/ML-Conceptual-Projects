#Load and Look
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data #(569, 30)
y = data.target # 0 = malignant, 1 = benign
feature_names = data.feature_names

print(X.shape, np.bincount(y))
print(pd.DataFrame(X, columns=feature_names).describe().T[['mean', 'std', 'min', 'max']])

# Standardize

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled.mean(axis=0).round(2)[:5]) #~0
print(X_scaled.std(axis=0).round(2)[:5]) #~1

# Fit PCA

from sklearn.decomposition import PCA

pca_full = PCA() # keep all 30 components

X_pca_full = pca_full.fit_transform(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

for i in range(10):
    print(f"PC{i+1}: {explained[i]*100:.2f}% {cumulative[i]*100:.2f}%)")

# Scree Plot + Cumulative Variance

fig, ax = plt.subplots(1,2, figsize = (12, 4))

ax[0].bar(range(1, 31), explained * 100)
ax[0].set_xlabel("Principal Component")
ax[0].set_ylabel("Explained Variance (%)")
ax[0].set_title("Scree Plot")

ax[1].plot(range(1, 31), cumulative * 100, marker='o')
ax[1].axhline(90, color='red', linestyle= '--', label= '90%')
ax[1].set_xlabel("Number of Components")
ax[1].set_ylabel("Cumulative Variance (%)")
ax[1].set_title("Cumulative Explained Variance")
ax[1].legend()

plt.tight_layout()
plt.show()

pca_2 = PCA(n_components=2)
X_2d = pca_2.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
for label, color, name in [(0, 'red', 'Manlignant'), (1, 'green', "Benign")]:
    plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1],
                c=color,label=name,alpha=0.6,edgecolor='k', s=30)

plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Breast Cancer -- PCA (2D)")
plt.legend()
plt.grid(alpha= 0.3)
plt.show()

# Interpretation of PCA Components

loadings = pd.DataFrame(
    pca_2.components_.T,
    columns=['PC1', 'PC2'],
    index= feature_names
)

print(loadings.sort_values('PC1', key=abs, ascending=False).head(10))
print(loadings.sort_values('PC2', key=abs, ascending=False).head(10))

#Biplot

plt.figure(figsize=(9, 7))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=25)

scale = 4  # arrows scaled for visibility
for i, feat in enumerate(feature_names):
    plt.arrow(0, 0,
              pca_2.components_[0, i] * scale,
              pca_2.components_[1, i] * scale,
              color='k', alpha=0.5, head_width=0.05)
    plt.text(pca_2.components_[0, i] * scale * 1.15,
             pca_2.components_[1, i] * scale * 1.15,
             feat, fontsize=7, alpha=0.8)

plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Biplot — PCA + Feature Loadings")
plt.grid(alpha=0.3)
plt.show()

#Reconstruction Error

pca_10 = PCA(n_components=10)
X_reduced = pca_10.fit_transform(X_scaled)
X_reconstructed = pca_10.inverse_transform(X_reduced)

recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction MSE with 10 PCs: {recon_error:.4f}")
print(f"Variance retained: {pca_10.explained_variance_ratio_.sum()*100:.2f}%")