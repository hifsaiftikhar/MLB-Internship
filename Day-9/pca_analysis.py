import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target  

# Scale the features before PCA - PCA is sensitive to feature scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA to reduce the 4 original features down to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original number of features:", X.shape[1])
print("Reduced number of components:", X_pca.shape[1])
print("\nVariance explained by each component:", pca.explained_variance_ratio_)
print(f"Total variance retained: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

# 3. Visualize the PCA-transformed data, colored by true species
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.title("Iris Data After PCA (2 Components), Colored by True Species")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Species (0=setosa, 1=versicolor, 2=virginica)")
plt.tight_layout()
plt.savefig("pca_visualization.png")
plt.close()

print("\nPCA visualization saved as pca_visualization.png")
