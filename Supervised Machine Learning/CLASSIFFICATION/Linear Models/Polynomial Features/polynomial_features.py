import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
y = (df['Churn'] == 'Yes').astype(int)

# ==========================================
# PART 1: VISUALIZING THE CURVE (2 Features)
# ==========================================
print("=== PART 1: 2D Visual (Tenure vs MonthlyCharges) ===")
X_2d = df[['tenure', 'MonthlyCharges']].values
X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, stratify=y, random_state=42)

# Linear Model (Degree 1)
linear_pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
linear_pipe.fit(X_train_2d, y_train)

# Polynomial Model (Degree 3)
poly_pipe = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LogisticRegression(class_weight='balanced')
)
poly_pipe.fit(X_train_2d, y_train)

# Plotting function
def plot_boundary(model, X, y, title, ax):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 50))
    
    # Predict probabilities on mesh
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot contour and scatter
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['green', 'red'])
    ax.scatter(X[y==0, 0], X[y==0, 1], label='No Churn', alpha=0.3, s=10, c='green')
    ax.scatter(X[y==1, 0], X[y==1, 1], label='Churn', alpha=0.3, s=10, c='red')
    ax.set_title(title)
    ax.set_xlabel('Tenure')
    ax.set_ylabel('Monthly Charges')
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_boundary(linear_pipe, X_test_2d, y_test, "Linear Boundary (Degree 1)", axes[0])
plot_boundary(poly_pipe, X_test_2d, y_test, "Polynomial Boundary (Degree 3)", axes[1])
plt.tight_layout()
plt.show()


# ==========================================
# PART 2: THE FEATURE EXPLOSION (All Features)
# ==========================================
print("\n=== PART 2: High-Dimensional Impact ===")
# Drop customerID first!
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

X_full = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)

print(f"Original features: {X_full.shape[1]}")

# Degree 2
poly2_pipe = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight='balanced')
)
poly2_pipe.fit(X_train, y_train)
poly2_features = poly2_pipe.named_steps['polynomialfeatures'].n_output_features_
f1_poly2 = f1_score(y_test, poly2_pipe.predict(X_test))

# Degree 3
poly3_pipe = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LogisticRegression(max_iter=3000, class_weight='balanced')
)
poly3_pipe.fit(X_train, y_train)
poly3_features = poly3_pipe.named_steps['polynomialfeatures'].n_output_features_
f1_poly3 = f1_score(y_test, poly3_pipe.predict(X_test))

# Baseline
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
baseline.fit(X_train, y_train)
f1_base = f1_score(y_test, baseline.predict(X_test))

print(f"Baseline (Degree 1): {X_full.shape[1]:4d} features | Test F1: {f1_base:.4f}")
print(f"Polynomial (Degree 2): {poly2_features:4d} features | Test F1: {f1_poly2:.4f}")
print(f"Polynomial (Degree 3): {poly3_features:4d} features | Test F1: {f1_poly3:.4f}")