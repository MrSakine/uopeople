import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, classification_report
from sklearn.impute import SimpleImputer

# ==========================================
# STEP 1: DATASET PREPROCESSING
# ==========================================
print("--- Step 1: Loading and Preprocessing Data ---")

# Load the dataset
df = pd.read_csv('Customer_Churn_Dataset_Final.csv')

# 1.1 Inspect initial data
print("Initial Data Shape:", df.shape)
print("\nMissing Values per column:\n", df.isnull().sum())

# 1.2 Drop irrelevant columns
# 'Customer ID' is a unique identifier and has no predictive power
if 'Customer ID' in df.columns:
    df = df.drop('Customer ID', axis=1)

# 1.3 Handle "Dirty" Data in Categorical Columns
# We will replace "5 (Unknown)" with the most frequent value (Mode) of that column.
mode_contract = df[df['Contract Type'] !=
                   '5 (Unknown)']['Contract Type'].mode()[0]
df['Contract Type'] = df['Contract Type'].replace('5 (Unknown)', mode_contract)
print(f"\nReplaced '5 (Unknown)' in Contract Type with mode: {mode_contract}")

# 1.4 Handle Missing Values in Numerical Columns
# 'Age', 'Monthly Charges', and 'Tenure (Months)' may have missing values (NaN)
imputer = SimpleImputer(strategy='median') # Using median to be robust against outliers
cols_to_impute = ['Age', 'Monthly Charges', 'Tenure (Months)']
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# 1.5 Handle Categorical Missing Values (if any exist beyond the specific case above)
cat_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns to check for missing values:", cat_cols.tolist())
for col in cat_cols:
    print(f"Missing values in {col}: {df[col].isnull().sum()}")
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after preprocessing:\n", df.isnull().sum())

# ==========================================
# STEP 2: FEATURE ENGINEERING
# ==========================================
print("\n--- Step 2: Feature Engineering ---")

# 2.1 Encoding Categorical Variables
le = LabelEncoder()

# Encode 'Churn' (Target): Yes -> 1, No -> 0
df['Churn'] = le.fit_transform(df['Churn'])

# Encode other categorical features
# Contract Type, Has Internet Service
categorical_features = ['Contract Type', 'Has Internet Service']
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

print("\nData after Encoding (First 5 rows):")
print(df.head())

# 2.2 Feature Scaling
# Gaussian Naive Bayes assumes features follow a normal distribution.
# Scaling helps the model treat features (like Tenure vs Monthly Charges) equally.
scaler = StandardScaler()
features_to_scale = ['Age', 'Monthly Charges', 'Tenure (Months)']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# ==========================================
# STEP 3: DATA SPLITTING
# ==========================================
print("\n--- Step 3: Data Splitting ---")

# Define Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into Training (80%) and Testing (20%) sets
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# ==========================================
# STEP 4: MODEL TRAINING
# ==========================================
print("\n--- Step 4: Model Training (Gaussian Naive Bayes) ---")

# Initialize the Gaussian Naive Bayes model
# We use Gaussian because our predictors (Age, Charges, Tenure) are continuous/scaled
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)
print("Model trained successfully.")

# ==========================================
# STEP 5: PERFORMANCE EVALUATION
# ==========================================
print("\n--- Step 5: Performance Evaluation ---")

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# 5.1 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix: Naive Bayes Churn Predictor')
plt.savefig('confusion_matrix.png')  # Saves the plot
plt.show()

# 5.2 Metrics Calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# STEP 6: VISUALIZATION (ROC CURVE)
# ==========================================
print("\n--- Step 6: ROC Curve and AUC ---")

# Get probability estimates for the positive class (Churn = Yes)
y_prob = nb_model.predict_proba(X_test)[:, 1]

# Calculate False Positive Rate (fpr), True Positive Rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate Area Under Curve (AUC)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
# Diagonal line (random guess)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')  # Saves the plot
plt.show()

print("\nAnalysis Complete. Please check the generated PNG files for visualizations.")
