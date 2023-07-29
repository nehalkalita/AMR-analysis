#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Given data for SA (Meropenem in µg/mL)
years_sa = np.array([2004, 2005, 2006, 2007, 2008, 2009, 2015, 2016, 2018, 2019, 2020, 2021])
mic_0to2_sa = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4, 67, 95, 93])
mic_4to8_sa = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
mic_gt16_sa = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
total_sa = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4, 67, 95, 93])

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
model_0to2_sa = LinearRegression()
model_0to2_sa.fit(years_sa.reshape(-1, 1), mic_0to2_sa)

model_4to8_sa = LinearRegression()
model_4to8_sa.fit(years_sa.reshape(-1, 1), mic_4to8_sa)

model_gt16_sa = LinearRegression()
model_gt16_sa.fit(years_sa.reshape(-1, 1), mic_gt16_sa)

model_total_sa = LinearRegression()
model_total_sa.fit(years_sa.reshape(-1, 1), total_sa)

model_0to2_mrsa = LinearRegression()
model_0to2_mrsa.fit(years_mrsa.reshape(-1, 1), mic_0to2_mrsa)

model_4_mrsa = LinearRegression()
model_4_mrsa.fit(years_mrsa.reshape(-1, 1), mic_4_mrsa)

model_gt8_mrsa = LinearRegression()
model_gt8_mrsa.fit(years_mrsa.reshape(-1, 1), mic_gt8_mrsa)

model_total_mrsa = LinearRegression()
model_total_mrsa.fit(years_mrsa.reshape(-1, 1), total_mrsa)

model_0to2_mssa = LinearRegression()
model_0to2_mssa.fit(years_mssa.reshape(-1, 1), mic_0to2_mssa)

model_4_mssa = LinearRegression()
model_4_mssa.fit(years_mssa.reshape(-1, 1), mic_4_mssa)

model_gt8_mssa = LinearRegression()
model_gt8_mssa.fit(years_mssa.reshape(-1, 1), mic_gt8_mssa)

model_total_mssa = LinearRegression()
model_total_mssa.fit(years_mssa.reshape(-1, 1), total_mssa)

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

# Calculate R^2 for each model (Interpolation)
r2_0to2_sa_interp = model_0to2_sa.score(years_sa.reshape(-1, 1), mic_0to2_sa)
r2_4to8_sa_interp = model_4to8_sa.score(years_sa.reshape(-1, 1), mic_4to8_sa)
r2_gt16_sa_interp = model_gt16_sa.score(years_sa.reshape(-1, 1), mic_gt16_sa)
r2_total_sa_interp = model_total_sa.score(years_sa.reshape(-1, 1), total_sa)

r2_0to2_mrsa_interp = model_0to2_mrsa.score(years_mrsa.reshape(-1, 1), mic_0to2_mrsa)
r2_4_mrsa_interp = model_4_mrsa.score(years_mrsa.reshape(-1, 1), mic_4_mrsa)
r2_gt8_mrsa_interp = model_gt8_mrsa.score(years_mrsa.reshape(-1, 1), mic_gt8_mrsa)
r2_total_mrsa_interp = model_total_mrsa.score(years_mrsa.reshape(-1, 1), total_mrsa)

r2_0to2_mssa_interp = model_0to2_mssa.score(years_mssa.reshape(-1, 1), mic_0to2_mssa)
r2_4_mssa_interp = model_4_mssa.score(years_mssa.reshape(-1, 1), mic_4_mssa)
r2_gt8_mssa_interp = model_gt8_mssa.score(years_mssa.reshape(-1, 1), mic_gt8_mssa)
r2_total_mssa_interp = model_total_mssa.score(years_mssa.reshape(-1, 1), total_mssa)

# Perform ARIMA modeling for each column
order = (1, 1, 1)  # ARIMA order (p, d, q)
arima_0to2_sa = ARIMA(mic_0to2_sa, order=order)
arima_0to2_sa_fit = arima_0to2_sa.fit()

arima_4to8_sa = ARIMA(mic_4to8_sa, order=order)
arima_4to8_sa_fit = arima_4to8_sa.fit()

arima_gt16_sa = ARIMA(mic_gt16_sa, order=order)
arima_gt16_sa_fit = arima_gt16_sa.fit()

arima_total_sa = ARIMA(total_sa, order=order)
arima_total_sa_fit = arima_total_sa.fit()

arima_0to2_mrsa = ARIMA(mic_0to2_mrsa, order=order)
arima_0to2_mrsa_fit = arima_0to2_mrsa.fit()

arima_4_mrsa = ARIMA(mic_4_mrsa, order=order)
arima_4_mrsa_fit = arima_4_mrsa.fit()

arima_gt8_mrsa = ARIMA(mic_gt8_mrsa, order=order)
arima_gt8_mrsa_fit = arima_gt8_mrsa.fit()

arima_total_mrsa = ARIMA(total_mrsa, order=order)
arima_total_mrsa_fit = arima_total_mrsa.fit()

arima_0to2_mssa = ARIMA(mic_0to2_mssa, order=order)
arima_0to2_mssa_fit = arima_0to2_mssa.fit()

arima_4_mssa = ARIMA(mic_4_mssa, order=order)
arima_4_mssa_fit = arima_4_mssa.fit()

arima_gt8_mssa = ARIMA(mic_gt8_mssa, order=order)
arima_gt8_mssa_fit = arima_gt8_mssa.fit()

arima_total_mssa = ARIMA(total_mssa, order=order)
arima_total_mssa_fit = arima_total_mssa.fit()

# Predict using ARIMA for year 2022
mic_0to2_sa_arima_pred = arima_0to2_sa_fit.forecast(steps=1)[0]
mic_4to8_sa_arima_pred = arima_4to8_sa_fit.forecast(steps=1)[0]
mic_gt16_sa_arima_pred = arima_gt16_sa_fit.forecast(steps=1)[0]
total_sa_arima_pred = arima_total_sa_fit.forecast(steps=1)[0]

mic_0to2_mrsa_arima_pred = arima_0to2_mrsa_fit.forecast(steps=1)[0]
mic_4_mrsa_arima_pred = arima_4_mrsa_fit.forecast(steps=1)[0]
mic_gt8_mrsa_arima_pred = arima_gt8_mrsa_fit.forecast(steps=1)[0]
total_mrsa_arima_pred = arima_total_mrsa_fit.forecast(steps=1)[0]

mic_0to2_mssa_arima_pred = arima_0to2_mssa_fit.forecast(steps=1)[0]
mic_4_mssa_arima_pred = arima_4_mssa_fit.forecast(steps=1)[0]
mic_gt8_mssa_arima_pred = arima_gt8_mssa_fit.forecast(steps=1)[0]
total_mssa_arima_pred = arima_total_mssa_fit.forecast(steps=1)[0]

# Calculate MSE for each model (ARIMA)
mse_0to2_sa_arima = mean_squared_error(np.array([112]), np.array([mic_0to2_sa_arima_pred]))
mse_4to8_sa_arima = mean_squared_error(np.array([0]), np.array([mic_4to8_sa_arima_pred]))
mse_gt16_sa_arima = mean_squared_error(np.array([0]), np.array([mic_gt16_sa_arima_pred]))
mse_total_sa_arima = mean_squared_error(np.array([112]), np.array([total_sa_arima_pred]))

mse_0to2_mrsa_arima = mean_squared_error(np.array([112]), np.array([mic_0to2_mrsa_arima_pred]))
mse_4_mrsa_arima = mean_squared_error(np.array([0]), np.array([mic_4_mrsa_arima_pred]))
mse_gt8_mrsa_arima = mean_squared_error(np.array([0]), np.array([mic_gt8_mrsa_arima_pred]))
mse_total_mrsa_arima = mean_squared_error(np.array([112]), np.array([total_mrsa_arima_pred]))

mse_0to2_mssa_arima = mean_squared_error(np.array([112]), np.array([mic_0to2_mssa_arima_pred]))
mse_4_mssa_arima = mean_squared_error(np.array([0]), np.array([mic_4_mssa_arima_pred]))
mse_gt8_mssa_arima = mean_squared_error(np.array([0]), np.array([mic_gt8_mssa_arima_pred]))
mse_total_mssa_arima = mean_squared_error(np.array([112]), np.array([total_mssa_arima_pred]))

# Print the results for SA
print("SA (Meropenem)")
print(f"Interpolation-based MIC (0 → 2) prediction for 2022: {mic_0to2_sa_interp_pred:.2f}, R^2: {r2_0to2_sa_interp:.2f}")
print(f"Interpolation-based MIC (4 → 8) prediction for 2022: {mic_4to8_sa_interp_pred:.2f}, R^2: {r2_4to8_sa_interp:.2f}")
print(f"Interpolation-based MIC (>= 16) prediction for 2022: {mic_gt16_sa_interp_pred:.2f}, R^2: {r2_gt16_sa_interp:.2f}")
print(f"Interpolation-based Total prediction for 2022: {total_sa_interp_pred:.2f}, R^2: {r2_total_sa_interp:.2f}")

print(f"ARIMA-based MIC (0 → 2) prediction for 2022: {mic_0to2_sa_arima_pred:.2f}, MSE: {mse_0to2_sa_arima:.2f}")
print(f"ARIMA-based MIC (4 → 8) prediction for 2022: {mic_4to8_sa_arima_pred:.2f}, MSE: {mse_4to8_sa_arima:.2f}")
print(f"ARIMA-based MIC (>= 16) prediction for 2022: {mic_gt16_sa_arima_pred:.2f}, MSE: {mse_gt16_sa_arima:.2f}")
print(f"ARIMA-based Total prediction for 2022: {total_sa_arima_pred:.2f}, MSE: {mse_total_sa_arima:.2f}")

# Print the results for MRSA
print("\nMRSA (Meropenem)")
print(f"Interpolation-based MIC (0 → 2) prediction for 2022: {mic_0to2_mrsa_interp_pred:.2f}, R^2: {r2_0to2_mrsa_interp:.2f}")
print(f"Interpolation-based MIC (4) prediction for 2022: {mic_4_mrsa_interp_pred:.2f}, R^2: {r2_4_mrsa_interp:.2f}")
print(f"Interpolation-based MIC (>= 8) prediction for 2022: {mic_gt8_mrsa_interp_pred:.2f}, R^2: {r2_gt8_mrsa_interp:.2f}")
print(f"Interpolation-based Total prediction for 2022: {total_mrsa_interp_pred:.2f}, R^2: {r2_total_mrsa_interp:.2f}")

print(f"ARIMA-based MIC (0 → 2) prediction for 2022: {mic_0to2_mrsa_arima_pred:.2f}, MSE: {mse_0to2_mrsa_arima:.2f}")
print(f"ARIMA-based MIC (4) prediction for 2022: {mic_4_mrsa_arima_pred:.2f}, MSE: {mse_4_mrsa_arima:.2f}")
print(f"ARIMA-based MIC (>= 8) prediction for 2022: {mic_gt8_mrsa_arima_pred:.2f}, MSE: {mse_gt8_mrsa_arima:.2f}")
print(f"ARIMA-based Total prediction for 2022: {total_mrsa_arima_pred:.2f}, MSE: {mse_total_mrsa_arima:.2f}")

# Print the results for MSSA
print("\nMSSA (Meropenem)")
print(f"Interpolation-based MIC (0 → 2) prediction for 2022: {mic_0to2_mssa_interp_pred:.2f}, R^2: {r2_0to2_mssa_interp:.2f}")
print(f"Interpolation-based MIC (4) prediction for 2022: {mic_4_mssa_interp_pred:.2f}, R^2: {r2_4_mssa_interp:.2f}")
print(f"Interpolation-based MIC (>= 8) prediction for 2022: {mic_gt8_mssa_interp_pred:.2f}, R^2: {r2_gt8_mssa_interp:.2f}")
print(f"Interpolation-based Total prediction for 2022: {total_mssa_interp_pred:.2f}, R^2: {r2_total_mssa_interp:.2f}")

print(f"ARIMA-based MIC (0 → 2) prediction for 2022: {mic_0to2_mssa_arima_pred:.2f}, MSE: {mse_0to2_mssa_arima:.2f}")
print(f"ARIMA-based MIC (4) prediction for 2022: {mic_4_mssa_arima_pred:.2f}, MSE: {mse_4_mssa_arima:.2f}")
print(f"ARIMA-based MIC (>= 8) prediction for 2022: {mic_gt8_mssa_arima_pred:.2f}, MSE: {mse_gt8_mssa_arima:.2f}")
print(f"ARIMA-based Total prediction for 2022: {total_mssa_arima_pred:.2f}, MSE: {mse_total_mssa_arima:.2f}")

import numpy as np
import matplotlib.pyplot as plt

# Data for plotting
categories = ['MIC (0 → 2)', 'MIC (4)', 'MIC (>= 8)']

# R^2 values for SA, MRSA, and MSSA (excluding Total column)
sa_interp_r2 = [r2_0to2_sa_interp, r2_4to8_sa_interp, r2_gt16_sa_interp]
mrsa_interp_r2 = [r2_0to2_mrsa_interp, r2_4_mrsa_interp, r2_gt8_mrsa_interp]
mssa_interp_r2 = [r2_0to2_mssa_interp, r2_4_mssa_interp, r2_gt8_mssa_interp]

# MSE values for SA, MRSA, and MSSA (excluding Total column)
sa_arima_mse = [mse_0to2_sa_arima, mse_4to8_sa_arima, mse_gt16_sa_arima]
mrsa_arima_mse = [mse_0to2_mrsa_arima, mse_4_mrsa_arima, mse_gt8_mrsa_arima]
mssa_arima_mse = [mse_0to2_mssa_arima, mse_4_mssa_arima, mse_gt8_mssa_arima]

# Create a function for plotting
def plot_comparison(categories, interp_r2, arima_mse, title):
    width = 0.35  # Width of the bars
    x = np.arange(len(categories))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, interp_r2, width, label='R^2 (Interpolation)')
    rects2 = ax.bar(x + width / 2, arima_mse, width, label='MSE (ARIMA)')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Performance Metric')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()

    # Add labels for each bar
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.show()

# Plot for SA (Vancomycin)
plot_comparison(categories, sa_interp_r2, sa_arima_mse, 'SA (Meropenem) - Interpolation vs. ARIMA')

# Plot for MRSA (Meropenem)
plot_comparison(categories, mrsa_interp_r2, mrsa_arima_mse, 'MRSA (Meropenem) - Interpolation vs. ARIMA')

# Plot for MSSA (Meropenem)
plot_comparison(categories, mssa_interp_r2, mssa_arima_mse, 'MSSA (Meropenem) - Interpolation vs. ARIMA')



########################PLOTTING GRAPHS

# Function to create curved comparison plot
def plot_curved_comparison(values1, values2, values3, labels1, labels2, labels3, title):
    plt.figure(figsize=(10, 6))
    plt.plot(categories, values1, marker='o', label=labels1)
    plt.plot(categories, values2, marker='o', label=labels2)
    plt.plot(categories, values3, marker='o', label=labels3)

    plt.xlabel('Categories')
    plt.ylabel('Performance Metric')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Add data points
    for i, value in enumerate(values1):
        plt.annotate(f'{value:.2f}',
                     xy=(categories[i], value),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom')

    for i, value in enumerate(values2):
        plt.annotate(f'{value:.2f}',
                     xy=(categories[i], value),
                     xytext=(0, -15),
                     textcoords="offset points",
                     ha='center', va='bottom')

    for i, value in enumerate(values3):
        plt.annotate(f'{value:.2f}',
                     xy=(categories[i], value),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.show()

# Plot R^2 comparison for SA, MRSA, and MSSA (excluding Total column)
plot_curved_comparison(sa_interp_r2, mrsa_interp_r2, mssa_interp_r2,
                       'SA (Meropenem)', 'MRSA (Meropenem)', 'MSSA (Meropenem)',
                       'Comparison of R^2 for 2022 - SA, MRSA, MSSA (Excluding Total)')

# Plot MSE comparison for SA, MRSA, and MSSA (excluding Total column)
plot_curved_comparison(sa_arima_mse, mrsa_arima_mse, mssa_arima_mse,
                       'SA (Meropenem)', 'MRSA (Meropenem)', 'MSSA (Meropenem)',
                       'Comparison of MSE for 2022 - SA, MRSA, MSSA (Excluding Total)')



# Data for plotting
categories = ['MIC (0 → 2)', 'MIC (4)', 'MIC (>= 8)']

# Values for SA (Meropenem) for 2022 (excluding the "Total" column)
sa_interp_values = [mic_0to2_sa_interp_pred, mic_4to8_sa_interp_pred, mic_gt16_sa_interp_pred]
sa_arima_values = [mic_0to2_sa_arima_pred, mic_4to8_sa_arima_pred, mic_gt16_sa_arima_pred]

# Values for MRSA (Meropenem) for 2022 (excluding the "Total" column)
mrsa_interp_values = [mic_0to2_mrsa_interp_pred, mic_4_mrsa_interp_pred, mic_gt8_mrsa_interp_pred]
mrsa_arima_values = [mic_0to2_mrsa_arima_pred, mic_4_mrsa_arima_pred, mic_gt8_mrsa_arima_pred]

# Values for MSSA (Meropenem) for 2022 (excluding the "Total" column)
mssa_interp_values = [mic_0to2_mssa_interp_pred, mic_4_mssa_interp_pred, mic_gt8_mssa_interp_pred]
mssa_arima_values = [mic_0to2_mssa_arima_pred, mic_4_mssa_arima_pred, mic_gt8_mssa_arima_pred]

# Function to create grouped bar plot
def plot_grouped_bar(categories, values1, values2, labels1, labels2, title):
    width = 0.35
    x = np.arange(len(categories))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, values1, width, label=labels1)
    rects2 = ax.bar(x + width / 2, values2, width, label=labels2)

    ax.set_xlabel('Categories')
    ax.set_ylabel('Values for 2022')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()

    # Add labels for each bar
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.show()

# Plot comparison for SA (Meropenem) - excluding Total column
plot_grouped_bar(categories, sa_interp_values, sa_arima_values, 'Interpolation', 'ARIMA', 'SA (Meropenem) - Interpolation vs. ARIMA')

# Plot comparison for MRSA (Meropenem) - excluding Total column
plot_grouped_bar(categories, mrsa_interp_values, mrsa_arima_values, 'Interpolation', 'ARIMA', 'MRSA (Meropenem) - Interpolation vs. ARIMA')

# Plot comparison for MSSA (Meropenem) - excluding Total column
plot_grouped_bar(categories, mssa_interp_values, mssa_arima_values, 'Interpolation', 'ARIMA', 'MSSA (Meropenem) - Interpolation vs. ARIMA')


# In[ ]:




