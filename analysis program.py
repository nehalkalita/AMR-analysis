"""#!/usr/bin/env python
# coding: utf-8
	Cefepime MIC (in µg/mL)			
Year	`0 → 8`	`16` `>= 32` Total
2018	11	0	4	15
2019	12	1	7	20
2020	8	0	9	17
2021	14	0	5	19
2022	?	?	?	?
2023	?	?	?	?
2024	?	?	?	?"""
# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Given data for Cefepime MIC (in µg/mL) for 0 → 8, 16, >= 32, and Total
years = np.array([2018, 2019, 2020, 2021])
mic_0to8_cefepime = np.array([12, 84, 27, 37])
mic_16_cefepime = np.array([3, 8, 2, 1])
mic_gt32_cefepime = np.array([18, 107, 44, 53])

# Function to train the ARIMA model and make predictions
def train_arima_model_and_predict(data, order):
    arima_model = ARIMA(data, order=order)
    arima_fit = arima_model.fit()
    prediction = arima_fit.forecast(steps=3)  # Forecasting for 3 years (2022, 2023, 2024)
    return prediction

# Perform ARIMA modeling for each column and calculate MSE
order = (1, 1, 1)  # ARIMA order (p, d, q)
mic_0to8_cefepime_arima_pred = train_arima_model_and_predict(mic_0to8_cefepime, order)
mic_16_cefepime_arima_pred = train_arima_model_and_predict(mic_16_cefepime, order)
mic_gt32_cefepime_arima_pred = train_arima_model_and_predict(mic_gt32_cefepime, order)

# Calculate MSE for ARIMA predictions
mse_0to8_cefepime_arima = mean_squared_error(mic_0to8_cefepime[-3:], mic_0to8_cefepime_arima_pred)
mse_16_cefepime_arima = mean_squared_error(mic_16_cefepime[-3:], mic_16_cefepime_arima_pred)
mse_gt32_cefepime_arima = mean_squared_error(mic_gt32_cefepime[-3:], mic_gt32_cefepime_arima_pred)

# Print the results for Cefepime MIC (0 → 8)
print("Cefepime MIC (0 → 8)")
#print(f"Interpolation-based MIC (0 → 8) predictions for 2022, 2023, and 2024: {mic_0to8_cefepime_interp_pred}")
print(f"ARIMA-based MIC (0 → 8) predictions for 2022, 2023, and 2024: {mic_0to8_cefepime_arima_pred}, MSE: {mse_0to8_cefepime_arima}")

# Print the results for Cefepime MIC (16)
print("\nCefepime MIC (16)")
#print(f"Interpolation-based MIC (16) predictions for 2022, 2023, and 2024: {mic_16_cefepime_interp_pred}")
print(f"ARIMA-based MIC (16) predictions for 2022, 2023, and 2024: {mic_16_cefepime_arima_pred}, MSE: {mse_16_cefepime_arima}")

# Print the results for Cefepime MIC (>= 32)
print("\nCefepime MIC (>= 32)")
#print(f"Interpolation-based MIC (>= 32) predictions for 2022, 2023, and 2024: {mic_gt32_cefepime_interp_pred}")
print(f"ARIMA-based MIC (>= 32) predictions for 2022, 2023, and 2024: {mic_gt32_cefepime_arima_pred}, MSE: {mse_gt32_cefepime_arima}")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Given data for Cefepime MIC (in µg/mL) for 0 → 8, 16, >= 32, and Total
years = np.array([2018, 2019, 2020, 2021])
mic_0to8_cefepime = np.array([12, 84, 27, 37])
mic_16_cefepime = np.array([3, 8, 2, 1])
mic_gt32_cefepime = np.array([18, 107, 44, 53])

# Function to train the ARIMA model and make predictions
def train_arima_model_and_predict(data, order):
    arima_model = ARIMA(data, order=order)
    arima_fit = arima_model.fit()
    prediction = arima_fit.forecast(steps=3)  # Forecasting for 3 years (2022, 2023, 2024)
    return prediction

# Perform ARIMA modeling for each column and calculate MSE
order = (1, 1, 1)  # ARIMA order (p, d, q)
mic_0to8_cefepime_arima_pred = train_arima_model_and_predict(mic_0to8_cefepime, order)
mic_16_cefepime_arima_pred = train_arima_model_and_predict(mic_16_cefepime, order)
mic_gt32_cefepime_arima_pred = train_arima_model_and_predict(mic_gt32_cefepime, order)

# Create a DataFrame to display the updated table
df = pd.DataFrame({
    'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    '0 → 8': np.concatenate((mic_0to8_cefepime, mic_0to8_cefepime_arima_pred)),
    '16': np.concatenate((mic_16_cefepime, mic_16_cefepime_arima_pred)),
    '>= 32': np.concatenate((mic_gt32_cefepime, mic_gt32_cefepime_arima_pred))
})

# Print the updated table
print(df)


# In[ ]:




