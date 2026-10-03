import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)
X = df.drop(columns=['Churn'])

# Identify categorical columns for CatBoost/LightGBM
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ---------- 2. Preprocessing for XGBoost ----------
# XGBoost needs One-Hot Encoding
X_xgb = pd.get_dummies(X, drop_first=True)
X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(
    X_xgb, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 3. Preprocessing for LightGBM and CatBoost ----------
# They can take raw DataFrames. We just need to cast objects to 'category' for LightGBM
X_lgb = X.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')

X_train_lgb, X_test_lgb, _, _ = train_test_split(
    X_lgb, y, test_size=0.2, stratify=y, random_state=42
)
# For CatBoost, we pass the original dataframe and just tell it which columns are cats
X_train_cat, X_test_cat, _, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 4. Train and Evaluate Models ----------
results = {}

# --- XGBoost ---
start = time.time()
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, 
                    scale_pos_weight=2.76, eval_metric='logloss', random_state=42)
xgb.fit(X_train_xgb, y_train)
xgb_time = time.time() - start
xgb_pred = xgb.predict(X_test_xgb)
results['XGBoost'] = {'F1': f1_score(y_test, xgb_pred), 'Time': xgb_time, 'Features': X_train_xgb.shape[1]}

# --- LightGBM ---
start = time.time()
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, 
                     scale_pos_weight=2.76, random_state=42, verbose=-1)
# LightGBM automatically handles the 'category' dtypes we set
lgb.fit(X_train_lgb, y_train)
lgb_time = time.time() - start
lgb_pred = lgb.predict(X_test_lgb)
results['LightGBM'] = {'F1': f1_score(y_test, lgb_pred), 'Time': lgb_time, 'Features': X_train_lgb.shape[1]}

# --- CatBoost ---
start = time.time()
cat = CatBoostClassifier(n_estimators=300, learning_rate=0.05, depth=5, 
                         scale_pos_weight=2.76, random_seed=42, verbose=0)
# We tell CatBoost exactly which columns are categorical
cat.fit(X_train_cat, y_train, cat_features=cat_cols)
cat_time = time.time() - start
cat_pred = cat.predict(X_test_cat)
results['CatBoost'] = {'F1': f1_score(y_test, cat_pred), 'Time': cat_time, 'Features': X_train_cat.shape[1]}

# ---------- 5. Print Results ----------
print(f"{'Model':<12} | {'F1 Score':<10} | {'Train Time':<12} | {'Input Features'}")
print("-" * 55)
for model_name, metrics in results.items():
    print(f"{model_name:<12} | {metrics['F1']:.4f}     | {metrics['Time']:.2f} sec     | {metrics['Features']}")

#CatBoost Feature Importance
print("\n--- Top 5 Features (CatBoost) ---")
feature_importances = cat.get_feature_importance()
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
print(feat_df.sort_values(by='Importance', ascending=False).head(5))