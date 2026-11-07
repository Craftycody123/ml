```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load dataset
digits = load_digits()
X = digits.data

# 2. Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Apply K-Means for different K values
inertia = []
silhouette = []
K_values = range(2, 11)

for k in K_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X, labels))

    print("K =", k)
    print("Inertia:", model.inertia_)
    print("Silhouette Score:", silhouette[-1])
    print()

# 4. Plot Inertia (Elbow Method)
plt.plot(K_values, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 5. Plot Silhouette Score
plt.plot(K_values, silhouette, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.show()
```