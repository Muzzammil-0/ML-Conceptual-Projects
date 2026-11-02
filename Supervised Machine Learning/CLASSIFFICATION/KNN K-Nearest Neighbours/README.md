# K-Nearest Neighbors (KNN) on the Sonar Dataset

##  Project Overview
This module explores the `KNeighborsClassifier` using the **Sonar dataset** (a classic binary classification problem). The objective is to predict whether an object is a **Mine (0)** or a **Rock (1)** based on 60 sonar frequency band energy measurements.

Instead of just training a model, this script acts as an experimental lab to understand the **Curse of Dimensionality**, the impact of **Distance Metrics**, and the effect of **PCA (Principal Component Analysis)** on a distance-based algorithm.

##  Dataset Details
*   **Source:** `sklearn.datasets.fetch_openml('sonar')`
*   **Samples:** 208
*   **Features:** 60 (numerical, continuous between 0.0 and 1.0)
*   **Target Classes:** Mine (111), Rock (97) — *Slightly imbalanced, but stratified.*

---

##  Key Concepts Explored

### 1. The "No Scaling" Anomaly
**Observation:** Raw accuracy and Scaled accuracy are identical (`0.7381`).
**Why?** KNN is highly sensitive to feature scaling. However, the Sonar dataset features are already normalized (all between 0.0 and 1.0). Therefore, `StandardScaler` (which forces mean=0, variance=1) doesn't drastically alter the relative distances between points in this specific dataset. If the features had been measured in different units (e.g., frequency in Hz vs. energy in Joules), scaling would have been critical.

### 2. Distance Metrics Matter
**Observation:** Manhattan and Cosine distance (`0.7619`) slightly outperform Euclidean distance (`0.7381`) on the raw 60D data.
**Why?** In high-dimensional spaces (60D), the Euclidean distance suffers heavily from the **Curse of Dimensionality** (distances between all points begin to look the same). Manhattan distance (L1 norm) is often more robust in high dimensions because it doesn't square the differences. Cosine distance focuses on the *angle* (orientation) rather than the *magnitude* of the vector, which is highly effective for frequency data like sonar signals.

### 3. Bias-Variance Tradeoff (The k-Sweep)
**Observation:** A plot is generated sweeping `k` from 1 to 30.
**Why?** 
*   **Low k (k=1):** High variance, low bias. It overfits to noise.
*   **High k (k=30):** Low variance, high bias. It smooths out the decision boundary too much.
*   The plot helps us visually identify the "sweet spot" where test accuracy peaks without overfitting.

### 4. The Curse of Dimensionality & PCA
**Observation:** PCA dramatically improves accuracy, peaking at **15 components (85.71% accuracy)** retaining 83.3% variance. 
**Why?** KNN calculates the distance between a test point and *every* training point. With 60 dimensions, many of those dimensions are just noise. PCA projects the data into a lower-dimensional space, discarding the noise and retaining the core variance. By reducing from 60D to 15D, KNN becomes significantly more accurate and computationally efficient.

---

## Results Summary

| Experiment | Accuracy | Takeaway |
| :--- | :--- | :--- |
| **Baseline (60D, k=5)** | 73.81% | Baseline performance. |
| **Raw (Euclidean)** | 73.81% | Standard metric. |
| **Raw (Manhattan)** | 76.19% | L1 norm handles high-dim noise better. |
| **Raw (Cosine)** | 76.19% | Angle-based similarity works well here. |
| **PCA to 15D (Euclidean)** | **85.71%** | 🏆 **Best Result.** Noise removal drastically boosts KNN. |
| **PCA to 15D (Manhattan)** | 80.95% | Decreased after PCA. L1 norm prefers the raw noise. |
| **PCA to 30D+** | 71.43% | Re-introducing noise hurts performance. |

*(Note: Test accuracy fluctuates because the dataset is very small (208 samples) and we are using an 80/20 split. Results may vary slightly depending on `random_state`.)*

---

