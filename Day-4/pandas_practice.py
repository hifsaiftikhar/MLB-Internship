import pandas as pd

# 1. Load the dataset
df = pd.read_csv("students.csv")

# 2. Display first and last five rows
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# 3. Display dataset information (column types, non-null counts)
print("\nDataset info:")
print(df.info())

# 4. Find missing values (True/False per cell, then total per column)
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Filter data based on a condition (students scoring above 80 in Math)
print("\nStudents scoring above 80 in Math:")
print(df[df["Math"] > 80])

# 6. Basic summary statistics (count, mean, std, min, max, etc.)
print("\nSummary statistics:")
print(df.describe())
