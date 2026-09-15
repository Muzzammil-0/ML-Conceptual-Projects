import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Suppress statsmodels warnings for clean output
warnings.filterwarnings('ignore')

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.index.freq = 'MS'

# ---------- 2. Chronological Split (80/20) ----------
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Get the boundary dates for slicing
train_end_date = train_df.index[-1]
test_start_date = test_df.index[0]

print(f"Train size: {len(train_df)} months | Test size: {len(test_df)} months\n")

# ---------- 3. MODEL 1: XGBoost (with Lags) ----------
df_xgb = df.copy()
df_xgb['lag_1'] = df_xgb['Passengers'].shift(1)
df_xgb['lag_12'] = df_xgb['Passengers'].shift(12)
df_xgb['rolling_mean_3'] = df_xgb['Passengers'].shift(1).rolling(window=3).mean()
df_xgb['month'] = df_xgb.index.month
df_xgb = df_xgb.dropna() # Drops the first 12 rows

features = ['lag_1', 'lag_12', 'rolling_mean_3', 'month']

# FIX: Slice by date boundaries instead of exact index matching
X_train_xgb = df_xgb[df_xgb.index <= train_end_date][features]
y_train_xgb = df_xgb[df_xgb.index <= train_end_date]['Passengers']

X_test_xgb = df_xgb[df_xgb.index >= test_start_date][features]
y_test = df_xgb[df_xgb.index >= test_start_date]['Passengers']

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb)
xgb_preds = xgb_model.predict(X_test_xgb)

# ---------- 4. MODEL 2: Vanilla ARIMA ----------
arima_model = ARIMA(train_df['Passengers'], order=(1, 1, 1))
arima_fitted = arima_model.fit()
arima_preds = arima_fitted.forecast(steps=len(test_df))

# ---------- 5. MODEL 3: Seasonal ARIMA (SARIMA) ----------
sarima_model = SARIMAX(train_df['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fitted = sarima_model.fit(disp=False)
sarima_preds = sarima_fitted.forecast(steps=len(test_df))

# ---------- 6. Evaluate All Models ----------
def eval_metrics(actual, pred, name):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    print(f"{name:<10} | RMSE: {rmse:6.2f} | MAPE: {mape:5.2f}%")

print("=== FINAL EVALUATION ===")
eval_metrics(y_test, xgb_preds, "XGBoost")
eval_metrics(y_test, arima_preds, "ARIMA")
eval_metrics(y_test, sarima_preds, "SARIMA")

# ---------- 7. Visualize the Forecasts ----------
plt.figure(figsize=(14, 7))
plt.plot(train_df.index, train_df['Passengers'], label='Train Data (History)', color='black', alpha=0.5)
plt.plot(test_df.index, y_test, label='Actual Future', color='green', linewidth=3)

plt.plot(test_df.index, xgb_preds, label='XGBoost (Lags)', color='red', linestyle='--')
plt.plot(test_df.index, arima_preds, label='ARIMA (1,1,1)', color='purple', linestyle='-.')
plt.plot(test_df.index, sarima_preds, label='SARIMA (Seasonal)', color='blue', linestyle=':')

plt.title('Airline Passenger Forecast: XGBoost vs ARIMA vs SARIMA')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()