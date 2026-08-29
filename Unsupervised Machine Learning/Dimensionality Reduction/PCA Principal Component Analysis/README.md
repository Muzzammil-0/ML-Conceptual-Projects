# PCA — Dimensionality Reduction on Breast Cancer Dataset

Principal Component Analysis applied to the Breast Cancer Wisconsin (Diagnostic) dataset to study variance structure, feature redundancy, and class separability in reduced space.

## Dataset
- 569 samples, 30 numeric features, 2 classes (malignant / benign)
- Loaded via `sklearn.datasets.load_breast_cancer`

## Approach
1. Standardize features (z-score) — required due to mixed scales across features
2. Fit full-spectrum PCA to analyze explained variance
3. Scree + cumulative variance plots to select components
4. 2D projection for visual inspection of class separation
5. Component loading analysis for interpretability
6. Reconstruction error check with 10 components

## Results
- 7 components retain **91%** of variance (30 → 7, ~4x compression)
- 10 components retain **95.16%**, reconstruction MSE = 0.0484
- PC1 (44.27%) — composite axis: concave points, concavity, compactness, perimeter, area
- PC2 (18.97%) — contrast axis: fractal dimension vs. size features
- Class separation is driven almost entirely by PC1; PC2 carries high variance but no discriminative signal

## Key Observation
Variance captured ≠ class relevance. PC2 holds 19% of the dataset variance yet contributes negligible separation between malignant and benign samples — a direct consequence of PCA being unsupervised.

## Stack
`Python` · `scikit-learn` · `numpy` · `pandas` · `matplotlib`

## Run
```bash
pip install numpy pandas matplotlib scikit-learn
python breast_cancer.py

^_~