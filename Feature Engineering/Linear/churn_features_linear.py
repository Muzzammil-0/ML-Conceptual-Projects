import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)

# ---------- 2. Prepare Datasets ----------
# Raw
X_raw = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

# Engineered
def engineer_features(data):
    df_eng = data.copy()
    df_eng['AvgMonthlyCharge'] = df_eng['TotalCharges'] / (df_eng['tenure'] + 1)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_eng['NumServices'] = df_eng[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df_eng['ChargeToTenureRatio'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
    df_eng['IsNewCustomer'] = (df_eng['tenure'] <= 3).astype(int)
    return df_eng

X_eng = pd.get_dummies(engineer_features(df).drop(columns=['Churn']), drop_first=True)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
X_train_eng, X_test_eng, _, _ = train_test_split(X_eng, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 3. Train Logistic Regression ----------
# We wrap it in a Pipeline with StandardScaler!
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

print("=== LOGISTIC REGRESSION (LINEAR MODEL) ===")

lr.fit(X_train_raw, y_train)
f1_raw = f1_score(y_test, lr.predict(X_test_raw))
print(f"Raw Features F1:        {f1_raw:.4f}")

lr.fit(X_train_eng, y_train)
f1_eng = f1_score(y_test, lr.predict(X_test_eng))
print(f"Engineered Features F1: {f1_eng:.4f}")