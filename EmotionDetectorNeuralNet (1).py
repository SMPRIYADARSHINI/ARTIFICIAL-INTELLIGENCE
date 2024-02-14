import pandas as pd
import numpy as np
import csv as csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load your training dataset
with open('trainYX.csv', mode='r') as t_file:
    train_reader = csv.reader(t_file)
    train_data = [list(map(float, row)) for row in train_reader]

X_train = np.array([row[1:] for row in train_data])
y_train = np.array([row[0] for row in train_data])

# Load your testing dataset
with open('testYX.csv', mode='r') as test_file:
    test_reader = csv.reader(test_file)
    test_data = [list(map(float, row)) for row in test_reader]

X_test = np.array([row[1:] for row in test_data])
y_test = np.array([row[0] for row in test_data])


# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but often recommended for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the MLP model
mlp_model1 = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500, random_state=42)
mlp_model2 = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)
mlp_model3 = MLPClassifier(hidden_layer_sizes=(150, ), max_iter=500, random_state=42)
mlp_model4 = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=500, random_state=42)
mlp_model5 = MLPClassifier(hidden_layer_sizes=(250, ), max_iter=500, random_state=42)


# Train the model
mlp_model1.fit(X_train, y_train)

# Predictions on the test set
y_pred = mlp_model1.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("hidden layer = ", 50, ", Accuracy: ", accuracy)

# Train the model
mlp_model2.fit(X_train, y_train)

# Predictions on the test set
y_pred = mlp_model2.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("hidden layer = ", 100, ", Accuracy: ", accuracy)

# Train the model
mlp_model3.fit(X_train, y_train)

# Predictions on the test set
y_pred = mlp_model3.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("hidden layer = ", 150, ", Accuracy: ", accuracy)

# Train the model
mlp_model4.fit(X_train, y_train)

# Predictions on the test set
y_pred = mlp_model4.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("hidden layer = ", 200, ", Accuracy: ", accuracy)

# Train the model
mlp_model5.fit(X_train, y_train)

# Predictions on the test set
y_pred = mlp_model5.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("hidden layer = ", 250, ", Accuracy: ", accuracy)
