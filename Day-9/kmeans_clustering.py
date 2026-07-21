import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
iris = load_iris()
X = iris.data  # features only - K-Means never sees the species labels

# Scale the features so no single feature (like petal length, which has a
# wider range) dominates the distance calculations K-Means relies on.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow Method: try a range of K values and record the inertia
# (inertia = sum of squared distances from each point to its cluster center;
# it always decreases as K increases, so we look for the point where adding
# more clusters stops giving a meaningful improvement - the "elbow").
inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia_values, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.close()
print("Elbow method chart saved as elbow_method.png")

# 3. Based on the elbow chart, K=3 is the chosen number of clusters
# (this matches the fact that Iris genuinely has 3 species).
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

print(f"\nApplied K-Means with K={optimal_k}")
print("Cluster sizes:")
print(pd.Series(clusters).value_counts().sort_index())

# 4. Visualize the clusters using two features (petal length and petal width,
# which separate the three Iris species most clearly)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 2], X[:, 3], c=clusters, cmap="viridis")
plt.title(f"K-Means Clusters (K={optimal_k}) - Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.close()
print("Cluster scatter plot saved as kmeans_clusters.png")
