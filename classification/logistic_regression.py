import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

# The script uses a Telco Churn, a hypothetical data file that concerns a telecommunications company's efforts to reduce
# turnover in its customer base. Each case corresponds to a separate customer, and it records various demographic and
# service usage information.


#  The dataset includes information about:
# Customers who left within the last month – the column is called Churn
# Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup,
# device protection, tech support, and streaming TV and movies.
# Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly
# charges, and total charges.
# Demographic info about customers – gender, age range, and if they have partners and dependents

# Load the dataset
churn_dataframe = pd.read_csv("../datasets/ChurnData.csv")

# Select relevant features for the model
churn_dataframe = churn_dataframe[
    ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
# Convert the 'churn' column to an integer type
churn_dataframe['churn'] = churn_dataframe['churn'].astype('int')

# Define feature set (X) and target variable (y)
X = np.asarray(churn_dataframe[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_dataframe['churn'])

# Normalize the feature set
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Create and train the logistic regression model
logistic_regression = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# Make predictions on the test set
yhat = logistic_regression.predict(X_test)

# Calculate predicted probabilities (not used further in this code)
yhat_probability = logistic_regression.predict_proba(X_test)

# Evaluate the model using Jaccard score
print(jaccard_score(y_test, yhat, pos_label=0))
