from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 3. Train a baseline Logistic Regression model (default settings, no tuning)
# max_iter is increased from the default of 100 because this dataset's features
# are on very different scales, which can prevent the model from converging
# in fewer iterations.
baseline_model = LogisticRegression(max_iter=5000)
baseline_model.fit(X_train, y_train)

# 4. Make predictions
y_pred = baseline_model.predict(X_test)

# 5. Evaluate the baseline model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nBaseline Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("(Rows = actual class, Columns = predicted class, order: [malignant, benign])")
