import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# This script reads GDP data of China from a CSV file, fits a sigmoid function to the data,
# and visualizes the original data along with the fitted curve.

# Define the sigmoid function
def sigmoid(input_data, beta_1, beta_2):
    """
    Sigmoid function used for curve fitting.

    Parameters:
    input_data (array): Input data
    beta_1 (float): Parameter for controlling the growth rate
    beta_2 (float): Parameter for controlling the inflection point

    Returns:
    y (array): Output of the sigmoid function
    """
    return 1 / (1 + np.exp(-beta_1 * (input_data - beta_2)))


# Load the dataset
dataframe = pd.read_csv("../datasets/china_gdp.csv")
years_data, gdp_data = dataframe["Year"].values, dataframe["Value"].values

# Normalize the data for better fitting
normalized_year_data = years_data / max(years_data)
normalized_gdp_data = gdp_data / max(gdp_data)

# Use curve_fit to find the best parameters for the sigmoid function
sigmoid_parameters, _ = curve_fit(sigmoid, normalized_year_data, normalized_gdp_data)

# Print the optimized parameters
print("Optimized parameters: beta_1 = %f, beta_2 = %f" % (sigmoid_parameters[0], sigmoid_parameters[1]))

# Generate years for prediction
years = np.linspace(1960, 2015, 55)
normalized_years = years / max(years)  # Normalize the years for prediction

# Generate prediction model for GDP using the optimized sigmoid function
predicted_gdp_model = sigmoid(normalized_years, *sigmoid_parameters)

# Plot original and predicted GDP
plt.figure(figsize=(8, 5))
plt.plot(normalized_year_data, normalized_gdp_data, 'ro', label='Data')
plt.plot(normalized_years, predicted_gdp_model, linewidth=3.0, label='Fit')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('Fitted Sigmoid Function vs Data')
plt.legend(loc='best')
plt.show()
