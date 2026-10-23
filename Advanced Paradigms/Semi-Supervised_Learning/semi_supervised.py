import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ---------- 1. Load Data ----------
data = load_breast_cancer()
X = data.data
y = data.target

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 2. Simulate the "Unlabeled" Problem ----------
# We will randomly hide 90% of the training labels by setting them to -1
np.random.seed(42)
n_train = len(y_train)
n_labeled = int(n_train * 0.10) # Keep only 10%

# Create a mask where True means "keep the label"
mask = np.zeros(n_train, dtype=bool)
mask[:n_labeled] = True
np.random.shuffle(mask)

y_train_semi = y_train.copy()
y_train_semi[~mask] = -1 # -1 is the required indicator for unlabeled data

print(f"Total training samples: {n_train}")
print(f"Labeled samples: {n_labeled} (10%)")
print(f"Unlabeled samples: {n_train - n_labeled} (90%)\n")

# ---------- 3. Baseline: Supervised on 10% Labels ----------
# We can only train on the data where y != -1
supervised_10 = RandomForestClassifier(n_estimators=100, random_state=42)
supervised_10.fit(X_train_scaled[mask], y_train[mask])
sup_10_acc = accuracy_score(y_test, supervised_10.predict(X_test_scaled))

# ---------- 4. Baseline: Supervised on 100% Labels (Oracle) ----------
supervised_100 = RandomForestClassifier(n_estimators=100, random_state=42)
supervised_100.fit(X_train_scaled, y_train)
sup_100_acc = accuracy_score(y_test, supervised_100.predict(X_test_scaled))

# ---------- 5. Semi-Supervised (Self-Training) ----------
# SelfTrainingClassifier trains on labeled data, predicts unlabeled data,
# takes the most confident predictions, adds them to the training set, and repeats.
base_model = RandomForestClassifier(n_estimators=100, random_state=42)

ssl_model = SelfTrainingClassifier(
    estimator=base_model,
    threshold=0.90,       # Only pseudo-label if the model is 90% confident
    criterion='threshold',
    max_iter=10,          # Maximum number of pseudo-labeling iterations
    verbose=True
)

# Fit the SSL model using the partially labeled data
ssl_model.fit(X_train_scaled, y_train_semi)
ssl_acc = accuracy_score(y_test, ssl_model.predict(X_test_scaled))

# ---------- 6. Compare Results ----------
print("\n=== FINAL RESULTS ===")
print(f"Supervised (10% labels):   {sup_10_acc:.4f}")
print(f"Semi-Supervised (10% + 90% unlabeled): {ssl_acc:.4f}")
print(f"Supervised (100% labels):  {sup_100_acc:.4f}  (Oracle Ceiling)")

print("\nSSL Classification Report:")
print(classification_report(y_test, ssl_model.predict(X_test_scaled), target_names=['Malignant', 'Benign']))