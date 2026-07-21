import pandas as pd
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()

# 2. Convert it into a Pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

print("Feature names:", iris.feature_names)
print("Species:", list(iris.target_names))

# Note: species/species_name are kept here only for later comparison against
# the clusters K-Means finds on its own. Unsupervised learning does not use
# these labels during clustering - K-Means never sees them.

# 3. Explore the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nSpecies distribution:")
print(df["species_name"].value_counts())
