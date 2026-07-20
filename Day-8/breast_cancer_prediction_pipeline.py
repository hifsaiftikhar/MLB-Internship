import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

print("===== BREAST CANCER PREDICTION SYSTEM =====\n")

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target classes: {list(data.target_names)} (0 = malignant, 1 = benign)\n")

# 2. Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features - fit only on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")

# 3. Train a baseline Logistic Regression model
baseline_model = LogisticRegression(max_iter=5000)
baseline_model.fit(X_train_scaled, y_train)
baseline_pred = baseline_model.predict(X_test_scaled)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

print("Baseline Model Evaluation:")
print(f"Accuracy: {baseline_accuracy:.4f} | Precision: {baseline_precision:.4f} | "
      f"Recall: {baseline_recall:.4f} | F1-Score: {baseline_f1:.4f}\n")

# 4. Hyperparameter tuning with GridSearchCV
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=5000),
    param_grid,
    cv=5,
    scoring="accuracy",
)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found by GridSearchCV:", grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}\n")

# 5. Evaluate the tuned model
tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test_scaled)

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_precision = precision_score(y_test, tuned_pred)
tuned_recall = recall_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred)

print("Tuned Model Evaluation:")
print(f"Accuracy: {tuned_accuracy:.4f} | Precision: {tuned_precision:.4f} | "
      f"Recall: {tuned_recall:.4f} | F1-Score: {tuned_f1:.4f}\n")

# 6. Compare baseline vs tuned
print("===== Baseline vs Tuned Comparison =====")
print(f"Accuracy:  Baseline {baseline_accuracy:.4f}  ->  Tuned {tuned_accuracy:.4f}")
print(f"F1-Score:  Baseline {baseline_f1:.4f}  ->  Tuned {tuned_f1:.4f}\n")

# 7. Confusion matrix heatmap comparing baseline and tuned models
baseline_cm = confusion_matrix(y_test, baseline_pred)
tuned_cm = confusion_matrix(y_test, tuned_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(baseline_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names, ax=axes[0])
axes[0].set_title("Baseline Model - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(tuned_cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=data.target_names, yticklabels=data.target_names, ax=axes[1])
axes[1].set_title("Tuned Model - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png")
plt.close()

print("Confusion matrix heatmap saved as confusion_matrix_comparison.png")
