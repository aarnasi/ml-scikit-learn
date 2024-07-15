import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Reading the dataset from the given URL
# This dataset contains various features of cars, including engine size and CO2 emissions
df = pd.read_csv('../datasets/fuel_comsumption_data.csv')

# Selecting specific columns for the analysis
cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]

# Splitting the data into training and testing sets using a random mask
mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]  # Training set (80% of the data)
test = cdf[~mask]  # Testing set (20% of the data)

# Creating a linear regression model
linear_regression = linear_model.LinearRegression()

# Preparing the training data
train_x = np.asanyarray(train[['ENGINESIZE']])  # Features
train_y = np.asanyarray(train[['CO2EMISSIONS']])  # Target variable

# Fitting the linear regression model on the training data
linear_regression.fit(train_x, train_y)

# Printing the coefficients and intercept of the trained model
print('Train: Coefficients:', linear_regression.coef_)
print('Train: Intercept:', linear_regression.intercept_)

# Plotting the fit line over the train data
plt.scatter(train_x, train_y, color='blue')
plt.plot(train_x, linear_regression.coef_[0][0] * train_x + linear_regression.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Preparing the testing data
test_x = np.asanyarray(test[['ENGINESIZE']])  # Features
test_y = np.asanyarray(test[['CO2EMISSIONS']])  # Target variable

# Predicting the CO2 emissions using the testing data
predict_y = linear_regression.predict(test_x)

# Evaluating the model performance on the testing data
print("Computing model performance")
print("Test:Mean aboslute error: %.2f" % np.mean(np.absolute(test_y - predict_y)))
print("Test: Residual sum of squares (MSE): %.2f" % np.mean(np.square(test_y - predict_y)))
print("Test: R2-score: %.2f" % r2_score(test_y, predict_y))
