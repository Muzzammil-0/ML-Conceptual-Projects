# Time Series Forecasting — Airline Passengers

Forecasting monthly airline passenger numbers using XGBoost with engineered lag features. Demonstrates chronological data splitting, feature engineering for time series, and the limitations of tree-based models on trending data.

## Dataset
- AirPassengers (monthly totals from 1949 to 1960)
- 144 rows, target variable: `Passengers`

## Approach
1. Convert the `Month` column to a `DatetimeIndex`.
2. Generate time-series features:
   - `lag_1`: Passengers from 1 month ago
   - `lag_12`: Passengers from 12 months ago (captures yearly seasonality)
   - `rolling_mean_3`: 3-month moving average
   - `month`: Extracted month number (1–12)
3. Drop the first 12 rows to account for the `NaN` values created by shifting.
4. Split data chronologically (first 80% train, last 20% test). **Do NOT shuffle.**
5. Train an XGBoost Regressor.
6. Evaluate using RMSE and MAPE (Mean Absolute Percentage Error).

## Results
- **RMSE:** 53.55 passengers
- **MAPE:** 8.97% (average error percentage)

*Note: The chronological split resulted in training on 105 months and testing on the final 27 months.*

## Key Observations
- **No Shuffling.** Standard `train_test_split` breaks time series data. You must train on the past and test on the future to prevent data leakage.
- **Lag Features are King.** Giving the model `lag_1` and `lag_12` tells it exactly what happened yesterday and last year at the same time. This is how ML models "learn" time.
- **MAPE over MSE.** Root Mean Squared Error (RMSE) is good for comparing models, but Mean Absolute Percentage Error (MAPE) is better for business context because it tells you the average error as a percentage (e.g., "Our forecast is off by 9%").
- **Tree Extrapolation Limit.** XGBoost struggles to predict trends that go higher than the training data. The model will often flatten out at the maximum value it saw during training. For purely linear trending data, statistical models like ARIMA or Linear Regression often outperform XGBoost in time series because they can mathematically extrapolate upward trends.

## Stack
`xgboost` · `pandas` · `numpy` · `matplotlib` · `scikit-learn`