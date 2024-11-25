import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Load data
data = pd.read_excel("Snowman_data.xlsx", sheet_name="Sheet1")
data['siq_ntrm_date'] = pd.to_datetime(data['siq_ntrm_date'])
data.set_index('siq_ntrm_date', inplace=True)

# Prepare target and exogenous variables
target_variable = data[['earnings']].dropna()
exog_variables = data[['total_income', 'change_in_stock', 'eps', 'liab', 'assets']]

# Fill missing values and handle infinities in exogenous variables
exog_variables = exog_variables.ffill().replace([np.inf, -np.inf], np.nan).dropna()

# Check for alignment in index between target and exogenous variables
target_variable = target_variable.loc[exog_variables.index]  # Align target variable with exogenous variable dates

# Set up SARIMAX model
model = SARIMAX(
    target_variable,
    exog=exog_variables,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 4)
)

# Fit and forecast with SARIMAX model
model_fit = model.fit(disp=False)
forecast_steps = 1  # Predicting next period
forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog_variables[-forecast_steps:])
forecast_ci = forecast.conf_int()

print("Predicted Earnings for Q2:", forecast.predicted_mean.iloc[-1])
print("Confidence Interval:", forecast_ci.iloc[-1])
