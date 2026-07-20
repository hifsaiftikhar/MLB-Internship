from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split into training and testing sets (same split as the baseline model,
# using the same random_state, so the comparison is fair)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Baseline model (for comparison)
baseline_model = LogisticRegression(max_iter=5000)
baseline_model.fit(X_train_scaled, y_train)
baseline_pred = baseline_model.predict(X_test_scaled)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

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
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# 5. Evaluate the tuned model on the test set
tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test_scaled)

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_precision = precision_score(y_test, tuned_pred)
tuned_recall = recall_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred)

print("\nTuned Model Evaluation (on test set):")
print(f"Accuracy: {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall: {tuned_recall:.4f}")
print(f"F1-Score: {tuned_f1:.4f}")

# 6. Compare baseline vs tuned
print("\n===== Baseline vs Tuned Comparison =====")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}   |   Tuned Accuracy: {tuned_accuracy:.4f}")
print(f"Baseline F1-Score: {baseline_f1:.4f}   |   Tuned F1-Score: {tuned_f1:.4f}")

if tuned_accuracy > baseline_accuracy:
    print("\nThe tuned model performed better than the baseline.")
elif tuned_accuracy < baseline_accuracy:
    print("\nThe tuned model performed worse than the baseline on this test set.")
else:
    print("\nThe tuned model performed the same as the baseline on this test set.")