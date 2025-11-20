import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


print("--- TASK 1: DATA EXPLORATION AND LOADING ---")

# a. Load the dataset and inspect its structure.
try:
    file_path = 'house_prices.csv'
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded from: {os.path.abspath(file_path)}")
except FileNotFoundError:
    print(
        f"ERROR: File '{file_path}' not found. Please ensure you have downloaded the CSV and saved it in the same directory.")
    exit()

# Inspecting the data structure
print("\nDataset Head (First 5 Rows):")
print(df.head())

print("\nDataset Information (Structure, types, and non-null counts):")
df.info()

df.columns = ['Size_sqft', 'Price_k']

# b. Plot a scatterplot of the data to visualize the relationship
# This code generates the plot. Save the resulting image as your .bmp/.jpg file.
plt.figure(figsize=(10, 6))
plt.scatter(df['Size_sqft'], df['Price_k'], color='darkblue', alpha=0.6)
plt.title('Relationship between House Size and Price ($ thousands)', fontsize=14)
plt.xlabel('House Size (Square Feet)', fontsize=12)
plt.ylabel('House Price ($ thousands)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('house_price_scatterplot.jpg')
plt.show()

print("\nScatterplot generated. Save the output image as your .bmp/.jpg file for submission.")

# ----------------------------------------------------------------------------------
print("\n--- TASK 2: MODEL BUILDING (SIMPLE LINEAR REGRESSION) ---")

# Prepare the data for scikit-learn
# Feature (X) must be a 2D array: reshape(-1, 1) converts the 1D series into a column vector
X = df[['Size_sqft']].values
y = df['Price_k'].values

# Split data into training and testing sets (80/20 split is standard practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# a. Implement a simple linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print(f"Model Training Complete.")
print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Coefficient (b1 - Price per sqft): {model.coef_[0]:.4f}")

# ----------------------------------------------------------------------------------
print("\n--- TASK 3: MODEL EVALUATION AND PREDICTION DEMO ---")

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# a. Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE) on the Test Set: {mse:.2f}")

# b. Demonstrate prediction for a new data point (e.g., 2126 sq ft)
house_size_to_predict = 2126
# The input must be a 2D array for the predict function
new_prediction = model.predict(np.array([[house_size_to_predict]]))
print(f"\nPrediction Demo (Size: {house_size_to_predict} sq ft):")
print(
    f"Predicted Price: ${new_prediction[0]:.2f} thousand (or ${new_prediction[0] * 1000:.2f})")

# Optional: Visualize the regression line on the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkblue', alpha=0.6, label='Actual Prices')
plt.plot(X, model.predict(X), color='red',
         linewidth=2, label='Regression Line')
# Highlight the predicted point
plt.scatter(house_size_to_predict, new_prediction[0], color='lime',
            s=100, edgecolors='black', zorder=5, label='Predicted Price')
plt.title('Simple Linear Regression Model Fit', fontsize=14)
plt.xlabel('House Size (Square Feet)', fontsize=12)
plt.ylabel('House Price ($ thousands)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
