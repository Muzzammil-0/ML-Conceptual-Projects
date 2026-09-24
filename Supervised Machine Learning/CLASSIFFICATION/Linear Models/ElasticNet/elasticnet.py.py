import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Ignore convergence warnings for the sake of clean output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------- 1. Load & Prepare Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Engineer redundant features (same as before)
df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargeToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)

y = (df['Churn'] == 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 2. Train the Models ----------
# We use solver='saga' because it supports elasticnet
# C is the overall regularization strength (we keep it fixed at 1.0)

models = {
    "Pure Lasso (L1)": LogisticRegression(l1_ratio=1.0, solver='saga', C=1.0, class_weight='balanced', random_state=42, max_iter=2000),
    "ElasticNet 50/50": LogisticRegression(l1_ratio=0.5, solver='saga', C=1.0, class_weight='balanced', random_state=42, max_iter=2000),
    "ElasticNet 80/20": LogisticRegression(l1_ratio=0.8, solver='saga', C=1.0, class_weight='balanced', random_state=42, max_iter=2000),
    "Pure Ridge (L2)": LogisticRegression(l1_ratio=0.0, solver='saga', C=1.0, class_weight='balanced', random_state=42, max_iter=2000)
}

print(f"Total features: {X.shape[1]}\n")
print("=== FEATURE SELECTION & COEFFICIENTS ===")

redundant_cols = ['AvgMonthlyCharge', 'ChargeToTenureRatio', 'TotalCharges', 'MonthlyCharges']

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    # Count non-zero weights
    non_zero = np.sum(model.coef_ != 0)
    
    # Get F1 score
    f1 = f1_score(y_test, model.predict(X_test_scaled))
    
    print(f"\n--- {name} ---")
    print(f"Features kept: {non_zero} / {X.shape[1]} | Test F1: {f1:.4f}")
    
    # Show the weights of the redundant features to observe the grouping effect
    print("Weights of redundant features:")
    for col in redundant_cols:
        idx = X.columns.get_loc(col)
        weight = model.coef_[0, idx]
        print(f"  {col:22s}: {weight:7.3f}")