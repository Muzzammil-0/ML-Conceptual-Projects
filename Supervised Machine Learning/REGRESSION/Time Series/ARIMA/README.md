# Time Series Forecasting — XGBoost vs. ARIMA vs. SARIMA

Comparing ML-based forecasting (XGBoost with engineered lag features) against classical statistical forecasting (ARIMA and SARIMA) on the AirPassengers dataset. Demonstrates chronological data splitting, feature engineering for time series, and the critical role of seasonality.

## Dataset
- AirPassengers (monthly totals from 1949 to 1960)
- 144 rows, target variable: `Passengers`

## Approach
1. Split data chronologically (first 80% train, last 20% test). **Do NOT shuffle.**
2. **XGBoost (ML Approach):** Engineered `lag_1`, `lag_12`, `rolling_mean_3`, and `month` features to teach the tree model about time and seasonality.
3. **ARIMA (Statistical):** Fitted a vanilla `ARIMA(1,1,1)` model. 
4. **SARIMA (Seasonal Statistical):** Fitted a `SARIMAX(1,1,1)(1,1,1,12)` model to explicitly capture the 12-month cycle.

## Results

| Model | RMSE | MAPE | Notes |
| :--- | :--- | :--- | :--- |
| **SARIMA (Seasonal)** | **30.14** | **5.05%** | **Winner.** Captures both upward trend and 12-month seasonality perfectly. |
| XGBoost (with lags) | 50.28 | 8.43% | Good seasonality, but flattens on the upward trend. |
| ARIMA (1,1,1) | 97.50 | 21.29% | Failed. Vanilla ARIMA has no concept of seasonality. |

## Key Observations
- **The Power of SARIMA:** Vanilla ARIMA completely missed the massive summer spikes, resulting in a 21% error rate. Adding the seasonal parameter `(1,1,1,12)` allowed SARIMA to mathematically model the 12-month cycle, slashing the error to 5%.
- **XGBoost requires manual seasonality:** Trees don't know what a "month" or a "year" is. XGBoost only performed well because we explicitly engineered `lag_12` and `month` features to hand-feed it the seasonality.
- **Tree Extrapolation Limit:** XGBoost predicts based on historical bounds. Because the trend is upward, it systematically under-predicts the future. SARIMA's `d` (differencing) parameter mathematically projects the upward slope into the future.
- **Data Leakage Prevention:** Standard `train_test_split` breaks time series data. You must train on the past and test on the future.
- **Pandas Frequency Warning:** When setting a DatetimeIndex for time series, explicitly set the frequency (`df.index.freq = 'MS'`) to prevent statsmodels from guessing incorrectly.

## Stack
`xgboost` · `statsmodels` · `pandas` · `numpy` · `matplotlib`