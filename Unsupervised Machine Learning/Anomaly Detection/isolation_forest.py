import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# ---------- 1. Create Synthetic Data ----------
# Generate 300 normal points in 2D
X_normal, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.60, random_state=42)

# Generate 20 anomalies (uniformly distributed, far away)
np.random.seed(42)
X_anomalies = np.random.uniform(low=-10, high=10, size=(20, 2))

# Combine them
X = np.vstack([X_normal, X_anomalies])

# ---------- 2. Train Isolation Forest ----------
# contamination = expected percentage of outliers in the dataset
# 20 anomalies out of 320 total = ~0.06
model = IsolationForest(contamination=0.06, random_state=42) #auto set contamination based on the proportion of anomalies
model.fit(X)

# Predict: 1 = Normal, -1 = Anomaly
predictions = model.predict(X)

# Get the anomaly score (lower score = more anomalous)
scores = model.decision_function(X)

# ---------- 3. Visualize the Results ----------
plt.figure(figsize=(10, 6))

# Plot Normal Points (1)
plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], 
            c='blue', label='Normal', alpha=0.5, edgecolor='k')

# Plot Anomalies (-1)
plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], 
            c='red', label='Anomaly', s=100, marker='X', edgecolor='k')

plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Print how many anomalies were found
print(f"Total points: {len(X)}")
print(f"Anomalies detected: {sum(predictions == -1)}")



#VERY non practical for real world data, but good for understanding the concept of isolation forest.