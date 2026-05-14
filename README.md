# Customer-Segmentation-Project
# ==========================================
# CUSTOMER SEGMENTATION PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# LOAD DATASET
# ==========================================

# Replace with your dataset file name
data = pd.read_csv("Mall_Customers.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(data.head())

# Dataset Information
print("\nDataset Info:")
print(data.info())

# Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# ==========================================
# DATA VISUALIZATION
# ==========================================

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.show()

# ==========================================
# FEATURE SELECTION
# ==========================================

# Select important columns
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# ==========================================
# DATA SCALING
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# ELBOW METHOD
# ==========================================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ==========================================
# APPLY K-MEANS CLUSTERING
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

# Predict Clusters
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ==========================================
# DISPLAY CLUSTERED DATA
# ==========================================

print("\nClustered Data:")
print(data.head())

# ==========================================
# VISUALIZE CLUSTERS
# ==========================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    data=data,
    s=100
)

# Plot Cluster Centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:,0],
    centers[:,1],
    c='black',
    s=300,
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# SAVE OUTPUT
# ==========================================

data.to_csv("Customer_Segmentation_Output.csv", index=False)

print("\nProject Completed Successfully!")
print("Clustered output saved as 'Customer_Segmentation_Output.csv'")