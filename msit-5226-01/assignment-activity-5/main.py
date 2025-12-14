# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("==========================================")
print("TASK 1: ANALYZING EXISTING CODE")
print("==========================================")

print("\n--- Task 1b: Original Code (hidden_layer_sizes=(10,)) ---")

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create a neural network classifier with 10 hidden nodes
mlp_10 = MLPClassifier(hidden_layer_sizes=(
    10,), max_iter=1000, random_state=42)

# Train the model
mlp_10.fit(X_train, y_train)

# Make predictions on the test set
y_pred_10 = mlp_10.predict(X_test)

# Evaluate the model
accuracy_10 = accuracy_score(y_test, y_pred_10)
print(f'Accuracy with 10 hidden nodes: {accuracy_10:.2f}')


print("\n--- Task 1c: Modified Code (hidden_layer_sizes=(30,)) ---")

# Create a neural network classifier with 30 hidden nodes
mlp_30 = MLPClassifier(hidden_layer_sizes=(
    30,), max_iter=1000, random_state=42)

# Train the model
mlp_30.fit(X_train, y_train)

# Make predictions
y_pred_30 = mlp_30.predict(X_test)

# Evaluate the model
accuracy_30 = accuracy_score(y_test, y_pred_30)
print(f'Accuracy with 30 hidden nodes: {accuracy_30:.2f}')


# --- TASK 1d: Print Predictions with Name ---
print("\n--- Task 1d: Printing Predictions with Name ---")
# Replace '[Your Name]' with your actual name in the line below
print(f"Result obtained by Fily Mohamed SAKINE:\n", y_pred_30)


print("\n\n==========================================")
print("TASK 2: WINE CLASSIFICATION")
print("==========================================")

# --- TASK 2a & 2b: Load Data and List Features ---
# Load the wine dataset
wine_data = load_wine()
X_wine = wine_data.data
y_wine = wine_data.target

print("\n--- Task 2b: Features and Target ---")
print("Feature Names:", wine_data.feature_names)
print("Target Classes:", wine_data.target_names)

# --- TASK 2c: Neural Network Implementation ---
print("\n--- Task 2c: Model Implementation & Results ---")

# 1. Split the data
# Using an 70-30 split for training and testing
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42
)

# 2. Data Preprocessing (Scaling)
# Neural Networks are sensitive to feature scaling. We use StandardScaler.
scaler = StandardScaler()
# Fit on training data only to prevent data leakage
scaler.fit(X_train_wine)
X_train_wine_scaled = scaler.transform(X_train_wine)
X_test_wine_scaled = scaler.transform(X_test_wine)

# 3. Initialize the MLPClassifier
# Using 3 hidden layers with sizes (10, 10, 10) for a deeper architecture
# max_iter is increased to allow the gradient descent to converge
mlp_wine = MLPClassifier(
    hidden_layer_sizes=(10, 10, 10),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

# 4. Train the model
mlp_wine.fit(X_train_wine_scaled, y_train_wine)

# 5. Predict on test data
y_pred_wine = mlp_wine.predict(X_test_wine_scaled)

# 6. Evaluate
acc_wine = accuracy_score(y_test_wine, y_pred_wine)
print(f"Wine Classification Accuracy: {acc_wine:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_wine, y_pred_wine))

print("\nClassification Report:")
print(classification_report(y_test_wine, y_pred_wine,
      target_names=wine_data.target_names))
