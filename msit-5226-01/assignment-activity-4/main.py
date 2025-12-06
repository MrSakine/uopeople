# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# ==========================================
# PART 5a: LOAD THE DATASET
# ==========================================
print("--- Loading Data ---")
try:
    df = pd.read_csv('wine_quality.csv')
    print("Dataset loaded successfully.")
    print(f"Data Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'wine_quality.csv' not found. Please download the dataset and save it locally.")
    exit()

# Filter for the selected parameters: pH, Alcohol, Sulfur Dioxide
# Note: Dataset usually contains 'total sulfur dioxide'. We will use that.
selected_features = ['pH', 'alcohol', 'total sulfur dioxide']

try:
    df_selected = df[selected_features].copy()
    print("\nSelected features for clustering:")
    print(df_selected.head())
except KeyError as e:
    print(f"\nError: Column not found in dataset. Check CSV headers. {e}")
    exit()

# ==========================================
# PART 5b: PREPROCESS DATA
# ==========================================
print("\n--- Preprocessing Data ---")

# 1. Handle Missing Values
# Using median imputation to be robust against outliers
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_selected), columns=selected_features)

# 2. Scale the Features
# K-Means relies on Euclidean distance, so scaling is mandatory.
# pH is small (3.0-4.0), Total SO2 is large (10-300). Without scaling, SO2 would dominate.
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# Convert back to DataFrame for easier handling later
df_scaled = pd.DataFrame(df_scaled, columns=selected_features)
print("Data scaled using StandardScaler.")

# ==========================================
# PART 5c: IMPLEMENT CLUSTERING & ELBOW METHOD
# ==========================================
print("\n--- Determining Optimal Clusters (Elbow Method) ---")

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.savefig('elbow_plot.jpg') # Save for report
plt.show()

# Based on typical wine datasets, k=3 or k=4 is usually optimal.
# We will select k=3 for this implementation (Adjust based on the Elbow plot you see).
optimal_k = 3
print(f"\nApplying K-Means with k={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(df_scaled)

# Add cluster labels back to the original (unscaled) dataframe for interpretation
df_selected['Cluster'] = clusters

# ==========================================
# PART 4 & 5c: SILHOUETTE SCORE
# ==========================================
sil_score = silhouette_score(df_scaled, clusters)
print(f"\nSilhouette Score for k={optimal_k}: {sil_score:.4f}")

# ==========================================
# PART 5d & 6: VISUALIZE CLUSTERS
# ==========================================
print("\n--- Visualizing Clusters ---")

# Pairplot to see relationships between pH, Alcohol, and SO2 colored by Cluster
# This satisfies the requirement for "visualizations of the clusters"
sns.pairplot(df_selected, hue='Cluster', palette='viridis', diag_kind='kde')
plt.suptitle(f'Wine Attributes Pairplot by Cluster (k={optimal_k})', y=1.02)
plt.savefig('cluster_pairplot.jpg') # Save as image file for submission
plt.show()

# ==========================================
# PART 7: CLUSTER SUMMARY
# ==========================================
print("\n--- Cluster Summary ---")
# Group by cluster to see the mean values of the attributes
cluster_summary = df_selected.groupby('Cluster').mean()
print(cluster_summary)

print("\nAnalysis Complete. Please refer to 'cluster_pairplot.jpg' for the visual submission.")