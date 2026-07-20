import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Load the dataset (built into Scikit-learn)
data = load_breast_cancer()

# 2. Convert it into a Pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Target classes:", list(data.target_names))
print("(0 = malignant, 1 = benign)\n")

# 3. Explore the data
print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# 4. Check the distribution of the target classes
print("\nTarget class distribution:")
print(df["target"].value_counts())
print("\nTarget class distribution (percentage):")
print(df["target"].value_counts(normalize=True) * 100)
