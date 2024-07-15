# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics

# The objective of the script is to build a classifier, to predict the category of the telecom customers
# using the demographic data such as region, age, marital status etc. The script uses K-Nearest classifier (KNN).
# The 'custcat' field consists of the possible values that correspond to the four customer groups:
#  1- Basic Service 2- E-Service 3- Plus Service 4- Total Service


# Load telecom customer data
dataframe = pd.read_csv("../datasets/telecom_customer_data.csv")

# Select features for the model and convert to a NumPy array
X = dataframe[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
               'reside']].values

# Standardize the features
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Select the target variable
y = dataframe['custcat'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Define the number of neighbors for KNN
K = 4

# Initialize and fit the K-Nearest Neighbors classifier
k_neighbor_classifier = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)

# Predict the target values for the test set
yhat = k_neighbor_classifier.predict(X_test)

# Evaluate the model's performance
print('Train set accuracy:', metrics.accuracy_score(y_train, k_neighbor_classifier.predict(X_train)))
print('Test set accuracy:', metrics.accuracy_score(y_test, yhat))