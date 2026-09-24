import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# ---------- 1. Load Data & Engineer Features ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Re-use the exact feature engineering from before to create redundancy
def engineer_features(data):
    df_eng = data.copy()
    df_eng['AvgMonthlyCharge'] = df_eng['TotalCharges'] / (df_eng['tenure'] + 1)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_eng['NumServices'] = df_eng[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df_eng['ChargeToTenureRatio'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
    df_eng['IsNewCustomer'] = (df_eng['tenure'] <= 3).astype(int)
    return df_eng

df_eng = engineer_features(df)
y = (df_eng['Churn'] == 'Yes').astype(int)
X = pd.get_dummies(df_eng.drop(columns=['Churn']), drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale (Mandatory for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 2. L2 (Ridge) Model ----------
# C is the inverse of regularization strength. Smaller C = stronger ratio.
l2_model = LogisticRegression(l1_ratio=0.0, C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
l2_model.fit(X_train_scaled, y_train)

# ---------- 3. L1 (Lasso) Model ----------
l1_model = LogisticRegression(l1_ratio=1.0, C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
l1_model.fit(X_train_scaled, y_train)

# ---------- 4. Compare Sparsity (Feature Selection) ----------
print("=== SPARSITY COMPARISON ===")
print(f"Total features: {X.shape[1]}")
print(f"L2 Non-zero coefficients: {np.sum(l2_model.coef_ != 0)}")
print(f"L1 Non-zero coefficients: {np.sum(l1_model.coef_ != 0)}")
print(f"L1 dropped: {X.shape[1] - np.sum(l1_model.coef_ != 0)} features completely.")

# Look specifically at the redundant features we engineered
eng_cols = ['AvgMonthlyCharge', 'ChargeToTenureRatio', 'TotalCharges', 'MonthlyCharges']
print("\nWeights for redundant continuous features:")
for col in eng_cols:
    idx = X.columns.get_loc(col)
    print(f"  {col:22s} | L2 weight: {l2_model.coef_[0, idx]:7.3f} | L1 weight: {l1_model.coef_[0, idx]:7.3f}")

# ---------- 5. Interpretability: Top Drivers of Churn ----------
print("\n=== TOP 10 CHURN DRIVERS (L1 MODEL) ===")
coefs = pd.Series(l1_model.coef_[0], index=X.columns)
# Sort by absolute weight to get the most impactful features
top_10 = coefs.abs().sort_values(ascending=False).head(10)
top_10_signed = coefs[top_10.index]

# Plot
plt.figure(figsize=(10, 6))
colors = ['red' if v > 0 else 'green' for v in top_10_signed]
top_10_signed.sort_values().plot(kind='barh', color=colors)
plt.axvline(0, color='black', linestyle='--')
plt.title("Top 10 Churn Drivers (L1 Logistic Regression)")
plt.xlabel("Weight (Log-Odds) -> Red = Increases Churn, Green = Decreases Churn")
plt.grid(alpha=0.3)
plt.show()

# ---------- 6. Performance ----------
print(f"\nL2 Test F1: {f1_score(y_test, l2_model.predict(X_test_scaled)):.4f}")
print(f"L1 Test F1: {f1_score(y_test, l1_model.predict(X_test_scaled)):.4f}")