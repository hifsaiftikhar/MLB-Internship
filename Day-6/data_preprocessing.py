import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load the dataset (already cleaned in Day-5, includes Average_Score)
df = pd.read_csv("cleaned_student_performance.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# 2. Handle categorical data
# Performance is a categorical column (Excellent, Good, Average, Needs Improvement).
# It is demonstrated here using Label Encoding, but note: Performance is calculated
# directly FROM Average_Score, which is our target variable. Using it as a feature
# would leak the answer into the model (data leakage), so it is encoded here only
# to demonstrate the technique, and is NOT used as a feature for training.
label_encoder = LabelEncoder()
df["Performance_Encoded"] = label_encoder.fit_transform(df["Performance"])
print("\nPerformance categories encoded as:")
print(df[["Performance", "Performance_Encoded"]].drop_duplicates())

# 3. Average_Score already exists from Day-5 cleaning, so no need to recreate it.

# 4. Select features (X) and target (y)
# Features: Math, Python, Statistics
# Target: Average_Score
# ML (Machine_Learning marks) is deliberately left out of the features, so the
# model has to genuinely learn the relationship rather than just recomputing
# an exact average from all four subjects.
features = ["Math", "Python", "Statistics"]
target = "Average_Score"

X = df[features]
y = df[target]

print("\nFeature columns:", features)
print("Target column:", target)

# 5. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# 6. Feature scaling (Standardization)
# Important: the scaler is fit ONLY on training data, then used to transform
# both train and test sets. Fitting on the full dataset (including test data)
# would leak information about the test set into training - a common
# data leakage mistake.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nSample of scaled training features (first 3 rows):")
print(X_train_scaled[:3])

print("\nPreprocessing complete.")
