import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score

# The script uses SVM (Support Vector Machines) to build and train a model using human cell records,
# and classify cells as either benign or malignant.

# Load the cancer dataset, which is publicly available from the UCI Machine Learning Repository. The dataset consists
# of several hundred human cell sample records, each containing values for a set of cell characteristics. The 'ID'
# field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in
# fields 'Clump' to 'Mit'. These values are graded from 1 to 10, with 1 being the closest to benign. The 'Class'
# field contains the diagnosis, as confirmed by separate medical procedures, indicating whether the samples are
# benign (value = 2) or malignant (value = 4).

# Load the dataset into a pandas DataFrame
cell_dataframe = pd.read_csv("../datasets/cell_samples.csv")

# Remove rows with non-numeric values in the 'BareNuc' column and convert the column to integer type
cell_dataframe = cell_dataframe[pd.to_numeric(cell_dataframe['BareNuc'], errors='coerce').notnull()]
cell_dataframe['BareNuc'] = cell_dataframe['BareNuc'].astype(int)

# Select the features for training (columns from 'Clump' to 'Mit')
feature_df = cell_dataframe[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)  # Convert the feature DataFrame to a numpy array

# Convert the 'Class' column to integer type and select it as the target variable
cell_dataframe['Class'] = cell_dataframe['Class'].astype(int)
y = np.asarray(cell_dataframe['Class'])  # Convert the target column to a numpy array

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Create and train the Support Vector Machine (SVM) model using the radial basis function (RBF) kernel
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Make predictions on the testing set
yhat = clf.predict(X_test)

# Calculate the F1 score of the model
f1 = f1_score(y_test, yhat, average='weighted')
print(f"F1 Score: {f1}")