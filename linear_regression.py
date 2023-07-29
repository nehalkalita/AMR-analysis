import numpy as np
from sklearn.linear_model import LinearRegression

# Known MIC values for the years 2004 to 2021
years = np.array([2004, 2005, 2006, 2007, 2008, 2009, 2015, 2016, 2018, 2019, 2020, 2021])
mic_a = np.array([17, 0, 15, 45, 13, 14, 25, 30, 23, 198, 223, 247])
mic_b = np.array([0, 0, 0, 5, 0, 1, 1, 1, 0, 0, 0, 0])
mic_c = np.array([0, 0, 0, 10, 8, 5, 1, 4, 0, 0, 0, 0])

# Create and train separate linear regression models for each category
model_a = LinearRegression()
model_a.fit(years.reshape(-1, 1), mic_a)

model_b = LinearRegression()
model_b.fit(years.reshape(-1, 1), mic_b)

model_c = LinearRegression()
model_c.fit(years.reshape(-1, 1), mic_c)

# Calculate R^2 for each model
r2_a = model_a.score(years.reshape(-1, 1), mic_a)
r2_b = model_b.score(years.reshape(-1, 1), mic_b)
r2_c = model_c.score(years.reshape(-1, 1), mic_c)

# Predict the MIC values for year 20__
for yr in range(2022,2025):
    mic_a_pred = model_a.predict([[yr]])[0]
    mic_b_pred = model_b.predict([[yr]])[0]
    mic_c_pred = model_c.predict([[yr]])[0]

    # Display the predicted MIC values and R^2 for 2022
    print(f"Predicted MIC B for {yr}: {mic_a_pred:.2f} µg/mL, R^2: {r2_a:.2f}")
    print(f"Predicted MIC B for {yr}: {mic_b_pred:.2f} µg/mL, R^2: {r2_b:.2f}")
    print(f"Predicted MIC C for {yr}: {mic_c_pred:.2f} µg/mL, R^2: {r2_c:.2f}")
    print()