import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("===== IRIS FLOWER CLUSTERING & VISUALIZATION =====\n")

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # true species labels - used only afterward, to check how well clustering matched them

# 2. Convert to a Pandas DataFrame and explore
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

print(f"Dataset loaded: {df.shape[0]} samples, {len(iris.feature_names)} features")
print("\nFirst 5 rows:")
print(df.head())
print("\nSpecies distribution:")
print(df["species_name"].value_counts())

# 3. Scale features before clustering and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to select the optimal number of clusters
inertia_values = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(k_range, inertia_values, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("mini_project_elbow_method.png")
plt.close()

# Based on the elbow curve, K=3 is chosen (also matches the known 3 species)
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)
df["cluster"] = clusters

print(f"\nApplied K-Means with K={optimal_k}")
print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())

# 5. Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA variance retained: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

# 6. Visualize: original data, K-Means clusters, and PCA-transformed data,
# side by side for direct comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original data (petal length vs petal width), colored by true species
scatter0 = axes[0].scatter(X[:, 2], X[:, 3], c=y, cmap="viridis")
axes[0].set_title("Original Data (True Species)")
axes[0].set_xlabel("Petal Length (cm)")
axes[0].set_ylabel("Petal Width (cm)")
fig.colorbar(scatter0, ax=axes[0])

# K-Means clusters (same features), colored by assigned cluster
scatter1 = axes[1].scatter(X[:, 2], X[:, 3], c=clusters, cmap="viridis")
axes[1].set_title(f"K-Means Clusters (K={optimal_k})")
axes[1].set_xlabel("Petal Length (cm)")
axes[1].set_ylabel("Petal Width (cm)")
fig.colorbar(scatter1, ax=axes[1])

# PCA-transformed data, colored by true species
scatter2 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
axes[2].set_title("PCA-Transformed Data (True Species)")
axes[2].set_xlabel("Principal Component 1")
axes[2].set_ylabel("Principal Component 2")
fig.colorbar(scatter2, ax=axes[2])

plt.tight_layout()
plt.savefig("mini_project_comparison.png")
plt.close()

print("\nComparison chart saved as mini_project_comparison.png")

# 7. Compare clusters against true species
comparison_table = pd.crosstab(df["species_name"], df["cluster"])
print("\nCluster vs True Species (crosstab):")
print(comparison_table)

print("\n===== OBSERVATIONS =====")
print(f"Number of clusters formed: {optimal_k}")
print("Setosa is almost perfectly separated into its own cluster, since it is")
print("clearly distinct in petal measurements. Versicolor and virginica overlap")
print("slightly, since their petal sizes are closer to each other, so a small")
print("number of points from these two species get grouped together or swapped.")
print(f"PCA retained {sum(pca.explained_variance_ratio_) * 100:.2f}% of the original variance")
print("using just 2 components instead of 4, making the data much easier to")
print("visualize on a single 2D plot while preserving most of the structure.")
