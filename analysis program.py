#!/usr/bin/env python
# coding: utf-8

# In[4]:


# modified program to make it generic. just replace the datset.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Define a function to train the linear regression model
def train_linear_regression_model(years, data):
    model = LinearRegression()
    model.fit(years.reshape(-1, 1), data)
    return model

# Define a function to train the ARIMA model and make predictions
def train_arima_model_and_predict(data, order):
    # Remove the first value from the data to make it consistent with the ARIMA requirement
    data = data[1:]
    arima_model = ARIMA(data, order=order)
    arima_fit = arima_model.fit()
    prediction = arima_fit.forecast(steps=1)[0]
    return prediction, arima_fit.resid

# Given data for SA (Meropenem in µg/mL)
years_sa = np.array([ 2018, 2019, 2020, 2021])
mic_0to2_sa = np.array([ 4, 67, 95, 93])
mic_4to8_sa = np.array([ 0, 0, 0, 0])
mic_gt16_sa = np.array([ 0, 0, 0, 0])
total_sa = np.array([ 4, 67, 95, 93])

# Given data for MRSA (Meropenem in µg/mL)
years_mrsa = np.array([2004, 2005, 2006, 2007, 2008, 2009, 2015, 2016, 2018, 2019, 2020, 2021])
mic_0to2_mrsa = np.array([8, 1, 8, 14, 8, 4, 22, 15, 5, 142, 133, 112])
mic_4_mrsa = np.array([0, 0, 0, 5, 0, 1, 1, 1, 0, 0, 0, 0])
mic_gt8_mrsa = np.array([0, 0, 0, 10, 8, 5, 1, 4, 0, 0, 0, 0])
total_mrsa = np.array([8, 1, 8, 29, 16, 10, 24, 20, 5, 142, 133, 112])

# Given data for MSSA (Meropenem in µg/mL)
years_mssa = np.array([2004, 2005, 2006, 2007, 2008, 2009, 2015, 2016, 2018, 2019, 2020, 2021])
mic_0to2_mssa = np.array([17, 0, 15, 43, 13, 14, 25, 30, 23, 198, 224, 247])
mic_4_mssa = np.array([0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
mic_gt8_mssa = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
total_mssa = np.array([17, 0, 15, 45, 13, 14, 25, 30, 23, 198, 224, 247])

# Create and train a linear regression model for interpolation for each column
model_0to2_sa = train_linear_regression_model(years_sa, mic_0to2_sa)
model_4to8_sa = train_linear_regression_model(years_sa, mic_4to8_sa)
model_gt16_sa = train_linear_regression_model(years_sa, mic_gt16_sa)
model_total_sa = train_linear_regression_model(years_sa, total_sa)

model_0to2_mrsa = train_linear_regression_model(years_mrsa, mic_0to2_mrsa)
model_4_mrsa = train_linear_regression_model(years_mrsa, mic_4_mrsa)
model_gt8_mrsa = train_linear_regression_model(years_mrsa, mic_gt8_mrsa)
model_total_mrsa = train_linear_regression_model(years_mrsa, total_mrsa)

model_0to2_mssa = train_linear_regression_model(years_mssa, mic_0to2_mssa)
model_4_mssa = train_linear_regression_model(years_mssa, mic_4_mssa)
model_gt8_mssa = train_linear_regression_model(years_mssa, mic_gt8_mssa)
model_total_mssa = train_linear_regression_model(years_mssa, total_mssa)

# Perform interpolation for year 2022 using linear regression for each column
mic_0to2_sa_interp_pred = model_0to2_sa.predict([[2022]])[0]
mic_4to8_sa_interp_pred = model_4to8_sa.predict([[2022]])[0]
mic_gt16_sa_interp_pred = model_gt16_sa.predict([[2022]])[0]
total_sa_interp_pred = model_total_sa.predict([[2022]])[0]

mic_0to2_mrsa_interp_pred = model_0to2_mrsa.predict([[2022]])[0]
mic_4_mrsa_interp_pred = model_4_mrsa.predict([[2022]])[0]
mic_gt8_mrsa_interp_pred = model_gt8_mrsa.predict([[2022]])[0]
total_mrsa_interp_pred = model_total_mrsa.predict([[2022]])[0]

mic_0to2_mssa_interp_pred = model_0to2_mssa.predict([[2022]])[0]
mic_4_mssa_interp_pred = model_4_mssa.predict([[2022]])[0]
mic_gt8_mssa_interp_pred = model_gt8_mssa.predict([[2022]])[0]
total_mssa_interp_pred = model_total_mssa.predict([[2022]])[0]

# Compute R-squared for interpolation for each category
r_squared_0to2_sa = model_0to2_sa.score(years_sa.reshape(-1, 1), mic_0to2_sa)
r_squared_4to8_sa = model_4to8_sa.score(years_sa.reshape(-1, 1), mic_4to8_sa)
r_squared_gt16_sa = model_gt16_sa.score(years_sa.reshape(-1, 1), mic_gt16_sa)
r_squared_total_sa = model_total_sa.score(years_sa.reshape(-1, 1), total_sa)

r_squared_0to2_mrsa = model_0to2_mrsa.score(years_mrsa.reshape(-1, 1), mic_0to2_mrsa)
r_squared_4_mrsa = model_4_mrsa.score(years_mrsa.reshape(-1, 1), mic_4_mrsa)
r_squared_gt8_mrsa = model_gt8_mrsa.score(years_mrsa.reshape(-1, 1), mic_gt8_mrsa)
r_squared_total_mrsa = model_total_mrsa.score(years_mrsa.reshape(-1, 1), total_mrsa)

r_squared_0to2_mssa = model_0to2_mssa.score(years_mssa.reshape(-1, 1), mic_0to2_mssa)
r_squared_4_mssa = model_4_mssa.score(years_mssa.reshape(-1, 1), mic_4_mssa)
r_squared_gt8_mssa = model_gt8_mssa.score(years_mssa.reshape(-1, 1), mic_gt8_mssa)
r_squared_total_mssa = model_total_mssa.score(years_mssa.reshape(-1, 1), total_mssa)


# Perform ARIMA modeling for each column and calculate RMSE
order = (1, 1, 1)  # ARIMA order (p, d, q)
mic_0to2_sa_arima_pred, resid_0to2_sa = train_arima_model_and_predict(mic_0to2_sa, order)
mic_4to8_sa_arima_pred, resid_4to8_sa = train_arima_model_and_predict(mic_4to8_sa, order)
mic_gt16_sa_arima_pred, resid_gt16_sa = train_arima_model_and_predict(mic_gt16_sa, order)
total_sa_arima_pred, resid_total_sa = train_arima_model_and_predict(total_sa, order)

mic_0to2_mrsa_arima_pred, resid_0to2_mrsa = train_arima_model_and_predict(mic_0to2_mrsa, order)
mic_4_mrsa_arima_pred, resid_4_mrsa = train_arima_model_and_predict(mic_4_mrsa, order)
mic_gt8_mrsa_arima_pred, resid_gt8_mrsa = train_arima_model_and_predict(mic_gt8_mrsa, order)
total_mrsa_arima_pred, resid_total_mrsa = train_arima_model_and_predict(total_mrsa, order)

mic_0to2_mssa_arima_pred, resid_0to2_mssa = train_arima_model_and_predict(mic_0to2_mssa, order)
mic_4_mssa_arima_pred, resid_4_mssa = train_arima_model_and_predict(mic_4_mssa, order)
mic_gt8_mssa_arima_pred, resid_gt8_mssa = train_arima_model_and_predict(mic_gt8_mssa, order)
total_mssa_arima_pred, resid_total_mssa = train_arima_model_and_predict(total_mssa, order)

# Calculate RMSE for ARIMA predictions
rmse_0to2_sa_arima = np.sqrt(mean_squared_error(mic_0to2_sa[1:], resid_0to2_sa))
rmse_4to8_sa_arima = np.sqrt(mean_squared_error(mic_4to8_sa[1:], resid_4to8_sa))
rmse_gt16_sa_arima = np.sqrt(mean_squared_error(mic_gt16_sa[1:], resid_gt16_sa))
rmse_total_sa_arima = np.sqrt(mean_squared_error(total_sa[1:], resid_total_sa))

rmse_0to2_mrsa_arima = np.sqrt(mean_squared_error(mic_0to2_mrsa[1:], resid_0to2_mrsa))
rmse_4_mrsa_arima = np.sqrt(mean_squared_error(mic_4_mrsa[1:], resid_4_mrsa))
rmse_gt8_mrsa_arima = np.sqrt(mean_squared_error(mic_gt8_mrsa[1:], resid_gt8_mrsa))
rmse_total_mrsa_arima = np.sqrt(mean_squared_error(total_mrsa[1:], resid_total_mrsa))

rmse_0to2_mssa_arima = np.sqrt(mean_squared_error(mic_0to2_mssa[1:], resid_0to2_mssa))
rmse_4_mssa_arima = np.sqrt(mean_squared_error(mic_4_mssa[1:], resid_4_mssa))
rmse_gt8_mssa_arima = np.sqrt(mean_squared_error(mic_gt8_mssa[1:], resid_gt8_mssa))
rmse_total_mssa_arima = np.sqrt(mean_squared_error(total_mssa[1:], resid_total_mssa))

# Print the results for SA
print("SA (Meropenem)")
print(f"Interpolation-based MIC (0 → 2) prediction for 2022: {mic_0to2_sa_interp_pred:.2f}, R-squared: {r_squared_0to2_sa:.2f}")
print(f"Interpolation-based MIC (4 → 8) prediction for 2022: {mic_4to8_sa_interp_pred:.2f}, R-squared: {r_squared_4to8_sa:.2f}")
print(f"Interpolation-based MIC (>= 16) prediction for 2022: {mic_gt16_sa_interp_pred:.2f}, R-squared: {r_squared_gt16_sa:.2f}")
print(f"Interpolation-based Total prediction for 2022: {total_sa_interp_pred:.2f}, R-squared: {r_squared_total_sa:.2f}")


print(f"ARIMA-based MIC (0 → 2) prediction for 2022: {mic_0to2_sa_arima_pred:.2f}, RMSE: {rmse_0to2_sa_arima:.2f}")
print(f"ARIMA-based MIC (4 → 8) prediction for 2022: {mic_4to8_sa_arima_pred:.2f}, RMSE: {rmse_4to8_sa_arima:.2f}")
print(f"ARIMA-based MIC (>= 16) prediction for 2022: {mic_gt16_sa_arima_pred:.2f}, RMSE: {rmse_gt16_sa_arima:.2f}")
print(f"ARIMA-based Total prediction for 2022: {total_sa_arima_pred:.2f}, RMSE: {rmse_total_sa_arima:.2f}")

# Print the results for MRSA
print("\nMRSA (Meropenem)")
print(f"Interpolation-based MIC (0 → 2) prediction for 2022: {mic_0to2_mrsa_interp_pred:.2f}, R-squared: {r_squared_0to2_mrsa:.2f}")
print(f"Interpolation-based MIC (4) prediction for 2022: {mic_4_mrsa_interp_pred:.2f}, R-squared: {r_squared_4_mrsa:.2f}")
print(f"Interpolation-based MIC (>= 8) prediction for 2022: {mic_gt8_mrsa_interp_pred:.2f}, R-squared: {r_squared_gt8_mrsa:.2f}")
print(f"Interpolation-based Total prediction for 2022: {total_mrsa_interp_pred:.2f}, R-squared: {r_squared_total_mrsa:.2f}")


print(f"ARIMA-based MIC (0 → 2) prediction for 2022: {mic_0to2_mrsa_arima_pred:.2f}, RMSE: {rmse_0to2_mrsa_arima:.2f}")
print(f"ARIMA-based MIC (4) prediction for 2022: {mic_4_mrsa_arima_pred:.2f}, RMSE: {rmse_4_mrsa_arima:.2f}")
print(f"ARIMA-based MIC (>= 8) prediction for 2022: {mic_gt8_mrsa_arima_pred:.2f}, RMSE: {rmse_gt8_mrsa_arima:.2f}")
print(f"ARIMA-based Total prediction for 2022: {total_mrsa_arima_pred:.2f}, RMSE: {rmse_total_mrsa_arima:.2f}")

# Print the results for MSSA
print("\nMSSA (Meropenem)")
print(f"Interpolation-based MIC (0 → 2) prediction for 2022: {mic_0to2_mssa_interp_pred:.2f}, R-squared: {r_squared_0to2_mssa:.2f}")
print(f"Interpolation-based MIC (4) prediction for 2022: {mic_4_mssa_interp_pred:.2f}, R-squared: {r_squared_4_mssa:.2f}")
print(f"Interpolation-based MIC (>= 8) prediction for 2022: {mic_gt8_mssa_interp_pred:.2f}, R-squared: {r_squared_gt8_mssa:.2f}")
print(f"Interpolation-based Total prediction for 2022: {total_mssa_interp_pred:.2f}, R-squared: {r_squared_total_mssa:.2f}")

print(f"ARIMA-based MIC (0 → 2) prediction for 2022: {mic_0to2_mssa_arima_pred:.2f}, RMSE: {rmse_0to2_mssa_arima:.2f}")
print(f"ARIMA-based MIC (4) prediction for 2022: {mic_4_mssa_arima_pred:.2f}, RMSE: {rmse_4_mssa_arima:.2f}")
print(f"ARIMA-based MIC (>= 8) prediction for 2022: {mic_gt8_mssa_arima_pred:.2f}, RMSE: {rmse_gt8_mssa_arima:.2f}")
print(f"ARIMA-based Total prediction for 2022: {total_mssa_arima_pred:.2f}, RMSE: {rmse_total_mssa_arima:.2f}")


# In[ ]:




