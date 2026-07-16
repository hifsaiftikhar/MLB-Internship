# Day-7: Classification and Model Evaluation

## What is Classification?
Classification is a type of Machine Learning task where the model learns to assign 
an input to one of several predefined categories. For example, given measurements 
of a flower, the model predicts which species it belongs to.

## Classification vs Regression
- **Regression** predicts a continuous number (e.g. a student's average score).
- **Classification** predicts a category or label (e.g. which flower species).

## What I worked on today

### 1. Classification Practice (classification_practice.py)
- Loaded the Iris dataset from Scikit-learn
- Split data into 80% training and 20% testing
- Trained a Logistic Regression model
- Evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

### 2. Iris Classification Mini Project (iris_classification.py)
- Loaded and explored the Iris dataset (150 samples, 3 classes, 4 features)
- Trained both Logistic Regression and Decision Tree models
- Compared performance of both models
- Printed sample predictions with actual vs predicted values
- Saved confusion matrices as an image

## Evaluation Metrics Used

| Metric | What it measures |
|--------|-----------------|
| Accuracy | Overall percentage of correct predictions |
| Precision | Of all predicted positives, how many were actually positive |
| Recall | Of all actual positives, how many were correctly predicted |
| F1-Score | Balance between Precision and Recall |
| Confusion Matrix | Full breakdown of correct and incorrect predictions per class |

## Model Performance

Both Logistic Regression and Decision Tree achieved perfect scores (1.0) on all 
metrics. This is expected — the Iris dataset is a clean, well-separated dataset 
specifically designed for learning classification. The three flower species are 
clearly distinguishable by their petal measurements, making it a relatively easy 
problem for both models.

## Observations
- Both models performed identically on this dataset and this train-test split
- Perfect accuracy on a small, clean dataset does not mean the model would 
  perform this well on real-world noisy data
- The confusion matrix confirms zero misclassifications across all three classes

## Files
- classification_practice.py: classification and evaluation practice
- iris_classification.py: full Iris classification mini project
- confusion_matrix.png: confusion matrix comparison of both models
- README.md: this file

## Author
Hifsa Iftikhar