import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("===== STUDENT SCORE PREDICTION SYSTEM =====\n")

# 1. Load the dataset
df = pd.read_csv("cleaned_student_performance.csv")
print(f"Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns\n")

# 2. Preprocess the data

# Encode the categorical Performance column, for demonstration only.
label_encoder = LabelEncoder()
df["Performance_Encoded"] = label_encoder.fit_transform(df["Performance"])

# Features and target
features = ["Math", "Python", "Statistics"]
target = "Average_Score"

X = df[features]
y = df[target]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling - fit only on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")

# 3. Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 4. Predict student average scores on the test set
y_pred = model.predict(X_test_scaled)

# 5. Display model evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}\n")

# 6. Comparison table: Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2)
})
print("Actual vs Predicted Average_Score:")
print(comparison)

# 7. Visualize prediction results: Actual vs Predicted scatter plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, color="royalblue", s=80, label="Predictions")

# Reference line showing perfect prediction (Actual == Predicted)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction")

plt.title("Actual vs Predicted Average_Score")
plt.xlabel("Actual Average_Score")
plt.ylabel("Predicted Average_Score")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nScatter plot saved as actual_vs_predicted.png")
