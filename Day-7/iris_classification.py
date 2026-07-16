import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# 1. Load and explore the dataset
iris = load_iris()
X = iris.data
y = iris.target

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION SYSTEM")
print("=" * 50)
print(f"\nTotal samples: {X.shape[0]}")
print(f"Features: {iris.feature_names}")
print(f"Target classes: {list(str(name) for name in iris.target_names)}")
print(f"\nSamples per class:")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {np.sum(y == i)} samples")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# 3. Helper function to evaluate and print metrics
def evaluate_model(name, y_test, y_pred):
    print(f"\n{'-' * 40}")
    print(f"Model: {name}")
    print(f"{'-' * 40}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 4. Train and evaluate Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
evaluate_model("Logistic Regression", y_test, lr_pred)

# 5. Train and evaluate Decision Tree (Bonus)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
evaluate_model("Decision Tree", y_test, dt_pred)

# 6. Sample predictions — show actual vs predicted for first 10 test samples
print(f"\n{'=' * 50}")
print("SAMPLE PREDICTIONS (first 10 test samples)")
print(f"{'=' * 50}")
print(f"{'#':<5} {'Actual':<20} {'Predicted':<20} {'Correct?'}")
print("-" * 55)
for i in range(10):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[lr_pred[i]]
    correct = "Yes" if actual == predicted else "No"
    print(f"{i+1:<5} {actual:<20} {predicted:<20} {correct}")

# 7. Save Confusion Matrix as image
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, lr_pred,
    display_labels=iris.target_names,
    ax=axes[0]
)
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test, dt_pred,
    display_labels=iris.target_names,
    ax=axes[1]
)
axes[1].set_title("Decision Tree")

plt.suptitle("Confusion Matrices — Iris Classification", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png")