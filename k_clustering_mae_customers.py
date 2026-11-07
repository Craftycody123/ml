import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
# Replace with your file path
data = pd.read_csv("Mall_Customers.csv")

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# Step 2: Preprocessing
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: K-Means Clustering
# -------------------------------
k = 5

kmeans = KMeans(n_clusters=k, random_state=0)
k_labels = kmeans.fit_predict(X_scaled)

k_inertia = kmeans.inertia_
k_silhouette = silhouette_score(X_scaled, k_labels)

# -------------------------------
# Step 4: Agglomerative Clustering
# -------------------------------
agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
a_labels = agg.fit_predict(X_scaled)

a_silhouette = silhouette_score(X_scaled, a_labels)

# -------------------------------
# Step 5: Visualization
# -------------------------------
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=k_labels)
plt.title("K-Means Clustering")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=a_labels)
plt.title("Agglomerative Clustering")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

# -------------------------------
# Step 6: Dendrogram
# -------------------------------
Z = linkage(X_scaled, method='ward')

plt.figure()
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# -------------------------------
# Step 7: Results
# -------------------------------
print("K-Means Inertia:", k_inertia)
print("K-Means Silhouette:", k_silhouette)
print("Agglomerative Silhouette:", a_silhouette)