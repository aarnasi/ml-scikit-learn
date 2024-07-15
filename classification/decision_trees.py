# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load dataset
data = pd.read_csv("../datasets/drug200.csv", delimiter=",")

# Select features and convert to a NumPy array
data_frame = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Encode categorical variables
# Encode 'Sex' feature
sex_label_encoder = preprocessing.LabelEncoder()
sex_label_encoder.fit(['F', 'M'])
data_frame[:, 1] = sex_label_encoder.transform(data_frame[:, 1])

# Encode 'BP' feature
bp_label_encoder = preprocessing.LabelEncoder()
bp_label_encoder.fit(['LOW', 'NORMAL', 'HIGH'])
data_frame[:, 2] = bp_label_encoder.transform(data_frame[:, 2])

# Encode 'Cholesterol' feature
cholesterol_label_encoder = preprocessing.LabelEncoder()
cholesterol_label_encoder.fit(['NORMAL', 'HIGH'])
data_frame[:, 3] = cholesterol_label_encoder.transform(data_frame[:, 3])

# Select the target variable
y = data["Drug"]

# Split data into training and test sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(data_frame, y, test_size=0.3, random_state=3)

# Initialize and train the Decision Tree Classifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

# Predict the target values for the test set
predicted_drug_tree = drugTree.predict(X_testset)

# Display the first 5 predictions and corresponding actual values
print("Predictions:", predicted_drug_tree[0:5])
print("Actual values:", y_testset.values[0:5])

# Evaluate the model's accuracy
print("DecisionTree Accuracy:", metrics.accuracy_score(y_testset, predicted_drug_tree))