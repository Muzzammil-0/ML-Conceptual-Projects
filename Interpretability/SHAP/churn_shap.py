import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------- 1. Load & Prep Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 2. Train XGBoost ----------
model = XGBClassifier(
    colsample_bytree=0.827, learning_rate=0.016, max_depth=4,
    n_estimators=317, subsample=0.780, scale_pos_weight=2.76,
    eval_metric='logloss', random_state=42
)
model.fit(X_train, y_train)


# ---------- 3. Initialize SHAP ----------
explainer = shap.TreeExplainer(model)


shap_values = explainer(X_test)

print(f"SHAP values calculated for {X_test.shape[0]} customers.")

# ---------- 4. Global Interpretability (Summary Plot) ----------
plt.figure(figsize=(10, 6))
plt.title("Global Feature Impact (SHAP Summary)")

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.show()

# ---------- 5. Local Interpretability (Waterfall Plot) ----------
probs = model.predict_proba(X_test)[:, 1]
high_risk_idx = np.argmax(probs) 

print(f"\nExplaining Customer Index {high_risk_idx} (Predicted Churn Prob: {probs[high_risk_idx]:.2f})")

plt.figure(figsize=(10, 6))
# This will now work because shap_values is an Explanation object
shap.plots.waterfall(shap_values[high_risk_idx], show=False)
plt.tight_layout()
plt.show()