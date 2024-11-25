# %%
import pandas as pd
import statsmodels.api as sm

# Sample DataFrame
data = pd.DataFrame({
    'date': pd.to_datetime([
        '2015-03-31', '2015-06-30', '2015-09-30', '2015-12-31',
        '2016-03-31', '2016-06-30', '2016-09-30', '2016-12-31',
        '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31',
        '2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31',
        '2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31',
        '2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31',
        '2021-03-31', '2021-06-30', '2021-09-30', '2021-12-31',
        '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31',
        '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31',
        '2024-03-31', '2024-06-30'
    ]),
    'total_income': [632.8, 605.8, 565.3, 571.5, 533.7, 499.0, 479.7, 478.3,
                     450.0, 479.7, 451.5, 499.5, 550.5, 560.2, 579.3, 605.8,
                     620.6, 629.2, 594.2, 594.6, 619.0, 563.8, 580.8, 611.3,
                     655.6, 665.8, 711.2, 747.7, 779.9, 884.7, 1096.5, 1108.1,
                     1162.3, 1299.7, 1255.5, 1252.2, 1294.0, 1410.0],
    'earnings': [142.5, 54.6, 28.7, 50.8, 71.3, 17.9, -83.2, -20.2,
                 36.2, -20.6, -36.5, 2.7, 18.8, 7.4, 12.1, 23.5,
                 54.2, -89.9, -14.0, -34.5, -11.7, 7.8, -17.3, 14.2,
                 -4.2, 6.0, 4.7, 8.3, -2.2, 18.9, 26.7, 37.4,
                 51.0, 33.8, 29.3, 42.5, 21.5, 17.9],
    'eps': [0.86, 0.33, 0.18, 0.30, 0.43, 0.11, -0.50, -0.12,
            0.22, -0.12, -0.22, 0.01, 0.12, 0.04, 0.07, 0.14,
            0.32, -0.54, -0.08, -0.21, -0.07, 0.05, -0.10, 0.09,
            -0.03, 0.04, 0.03, 0.05, -0.01, 0.11, 0.16, 0.22,
            0.31, 0.20, 0.18, 0.25, 0.13, 0.11]
})

# Add time-based features
data['time_index'] = range(len(data))

# Prepare features for regression
X = data[['time_index', 'total_income']]
y_earnings = data['earnings']
y_eps = data['eps']

# Fit models using statsmodels
X = sm.add_constant(X)
model_earnings = sm.OLS(y_earnings, X).fit()
model_eps = sm.OLS(y_eps, X).fit()

# Adjust total income for the next quarter to target earnings close to 20
next_time_index = data['time_index'].max() + 1
next_total_income = 1230  # Further adjusted to bring earnings close to 20

# New data for prediction
X_extrapolate = pd.DataFrame({
    'const': [1],  # Add constant term
    'time_index': [next_time_index],
    'total_income': [next_total_income]
})

# Predictions
predicted_earnings = model_earnings.predict(X_extrapolate)[0]
predicted_eps = model_eps.predict(X_extrapolate)[0]

# Print the results
print("Extrapolated Values for September 30, 2024:")
print(f"Earnings: Approximately {predicted_earnings:.2f}")
print(f"Earnings Per Share (EPS): Approximately {predicted_eps:.2f}")

# %%



