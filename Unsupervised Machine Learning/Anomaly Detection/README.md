# Anomaly Detection — Isolation Forest

Using an Isolation Forest to detect anomalies in a dataset without labels, demonstrating the concept of isolation via random partitioning.

## Dataset
- Synthetic 2D dataset generated via `make_blobs`
- 300 normal points (2 dense clusters) + 20 injected uniform anomalies.

## Approach
1. Generate normal data clusters and inject outliers.
2. Train an `IsolationForest` with `contamination=0.06` (expecting ~6% anomalies).
3. Predict labels (1 for normal, -1 for anomaly) and extract anomaly scores via `decision_function`.
4. Visualize normal clusters vs. detected anomalies.

## Results
- Successfully identified the 20 injected anomalies. 
- Normal points inside the dense blobs received high (positive) scores, while points far outside the blobs received low (negative) scores.

## Key Observations
- **Isolation, not clustering.** Isolation Forest doesn't calculate distances to cluster centers like K-Means. It builds random trees and measures how many splits it takes to isolate a point. Anomalies are isolated quickly (short path length).
- **The `contamination` parameter.** This tells the model the expected proportion of outliers. If set too high, the model will flag normal points as anomalies to fill the quota. If set too low, it will miss real anomalies.
- **Anomaly scores vs. Hard labels.** While `.predict()` gives a binary label (-1 or 1), the `.decision_function()` provides a continuous score. In production, it's often better to rank by this score and send the top 100 most anomalous points for human review.
- **Unsupervised advantage.** We never told the model what an anomaly looks like. It learned the "shape of normal" purely from the density of the data.

## Stack
`scikit-learn` · `numpy` · `matplotlib`