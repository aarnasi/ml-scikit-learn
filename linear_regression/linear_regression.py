import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# Reading the dataset from the given URL
# This dataset contains various features of cars, including engine size and CO2 emissions
df = pd.read_csv(
    'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs'
    '/FuelConsumptionCo2.csv')

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

# Preparing the testing data
test_x = np.asanyarray(test[['ENGINESIZE']])  # Features
test_y = np.asanyarray(test[['CO2EMISSIONS']])  # Target variable

# Predicting the CO2 emissions using the testing data
predict_y = linear_regression.predict(test_x)

# Evaluating the model performance on the testing data
print("Test:Mean aboslute error: %.2f" % np.mean(np.absolute(test_y - predict_y)))
print("Test: Residual sum of squares (MSE): %.2f" % np.mean(np.square(test_y - predict_y)))
print("Test: R2-score: %.2f" % r2_score(test_y, predict_y))
