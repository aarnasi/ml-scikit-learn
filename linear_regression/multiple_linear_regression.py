import pandas as pd
import numpy as np
from sklearn import linear_model

# Predicts CO2 emissions (CO2EMISSIONS) of car based on ENGINESIZE,CYLINDERS & FUELCONSUMPTION_COMB fields from a
# dataset ; a CSV file that is related to fuel consumption and carbon dioxide emission of cars.
# Dataset is split into training and test sets, a linear regression model is created using training set.
# Linear model is evaluated using the test set, and finally model is used to predict CO2EMISSIONS.


# Read the data from the CSV file
dataframe = pd.read_csv("../FuelConsumption.csv")

# Select the relevant columns for the analysis
subset_data_frame = dataframe[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Split the data into training and testing sets (80% training, 20% testing)
mask = np.random.rand(len(dataframe)) < 0.8
train = subset_data_frame[mask]
test = subset_data_frame[~mask]

# Initialize the linear regression model
linear_regression = linear_model.LinearRegression()

# Prepare the training data
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

# Train the model
linear_regression.fit(x_train, y_train)

# Output the coefficients of the model
print('Coefficients:', linear_regression.coef_)

# Make predictions using the testing set
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = linear_regression.predict(x_test)

# Calculate and print the residual sum of squares
residual_sum_of_squares = np.mean((y_hat - y_test) ** 2)
print("Residual sum of squares: %.2f" % residual_sum_of_squares)

# Calculate and print the variance score: 1 is perfect prediction
variance_score = linear_regression.score(x_test, y_test)
print('Variance score: %.2f' % variance_score)