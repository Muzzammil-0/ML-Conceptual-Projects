import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ---------- 1. Load ----------
sonar = fetch_openml('sonar', version=1, as_frame=True)
X = sonar.data.values
y = LabelEncoder().fit_transform(sonar.target)   # Mine=0, Rock=1
print("Shape:", X.shape, "| Classes:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 2. Baseline: no scaling ----------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn.predict(X_test))
print(f"\n[Raw, no scaling]     acc = {acc_raw:.4f}")

# ---------- 3. With standardization ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn.fit(X_train_s, y_train)
acc_scaled = accuracy_score(y_test, knn.predict(X_test_s))
print(f"[Scaled]              acc = {acc_scaled:.4f}")

# ---------- 4. k-sweep (bias-variance) ----------
k_range = range(1, 31)
train_accs, test_accs = [], []
for k in k_range:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
    train_accs.append(accuracy_score(y_train, m.predict(X_train_s)))
    test_accs.append(accuracy_score(y_test,  m.predict(X_test_s)))

plt.figure(figsize=(8, 5))
plt.plot(k_range, train_accs, marker='o', label='Train')
plt.plot(k_range, test_accs,  marker='s', label='Test')
plt.xlabel("k (neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN — k sweep (Sonar, scaled)")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# ---------- 5. Metric comparison on raw 60D ----------
print("\n-- Metric comparison on raw 60 features --")
for metric in ['euclidean', 'manhattan', 'cosine']:
    m = KNeighborsClassifier(n_neighbors=5, metric=metric).fit(X_train_s, y_train)
    print(f"  {metric:10s} acc = {accuracy_score(y_test, m.predict(X_test_s)):.4f}")

# ---------- 6. PCA sweep ----------
print("\n-- KNN accuracy vs #PCA components --")
components = [2, 5, 10, 15, 20, 30, 40, 50]
accs = []
for n in components:
    pca = PCA(n_components=n).fit(X_train_s)
    Xtr = pca.transform(X_train_s)
    Xte = pca.transform(X_test_s)
    m = KNeighborsClassifier(n_neighbors=5).fit(Xtr, y_train)
    acc = accuracy_score(y_test, m.predict(Xte))
    accs.append(acc)
    print(f"  {n:3d} comps -> acc = {acc:.4f}  (var retained {pca.explained_variance_ratio_.sum()*100:.1f}%)")

plt.figure(figsize=(8, 5))
plt.plot(components, accs, marker='o')
plt.xlabel("# PCA components")
plt.ylabel("Test accuracy")
plt.title("KNN accuracy vs PCA dimensionality (Sonar)")
plt.grid(alpha=0.3); plt.show()

# ---------- 7. Metric comparison after PCA ----------
print("\n-- Metric comparison after PCA to 15D --")
pca = PCA(n_components=15).fit(X_train_s)
Xtr = pca.transform(X_train_s); Xte = pca.transform(X_test_s)
for metric in ['euclidean', 'manhattan', 'cosine']:
    m = KNeighborsClassifier(n_neighbors=5, metric=metric).fit(Xtr, y_train)
    print(f"  {metric:10s} acc = {accuracy_score(y_test, m.predict(Xte)):.4f}")