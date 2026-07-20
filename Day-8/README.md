# Day-8: Model Evaluation and Hyperparameter Tuning

## What I worked on

Today's focus was learning how to properly evaluate a Machine Learning model and improve it using hyperparameter tuning, using the Breast Cancer Wisconsin dataset built into Scikit-learn. This included building a baseline Logistic Regression model, tuning it with GridSearchCV, and comparing performance before and after tuning.

## Dataset exploration (dataset_exploration.py)

- Loaded the dataset directly from sklearn.datasets and converted it into a Pandas DataFrame.
- Explored it using .head(), .info(), and .describe().
- Checked the target class distribution: 357 benign cases (62.7%) and 212 malignant cases (37.3%) - a mild class imbalance worth being aware of, since a model could get misleadingly high accuracy just by favoring the majority class.

## Baseline model (baseline_model.py)

- Split the data into 80% training and 20% testing.
- Scaled the features using StandardScaler, fit only on training data, since this dataset's features are on very different scales (for example, "mean area" is in the hundreds, while "mean smoothness" is under 1).
- Trained a Logistic Regression model with default settings (no tuning).
- Evaluated using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

Baseline results: Accuracy 0.9737, Precision 0.9722, Recall 0.9859, F1-Score 0.9790.

## Hyperparameter tuning (hyperparameter_tuning.py)

Used GridSearchCV to search over a grid of Logistic Regression hyperparameters:
- C (regularization strength): 0.01, 0.1, 1, 10, 100
- solver: liblinear, lbfgs

GridSearchCV tests every combination using 5-fold cross-validation and selects the combination with the best average performance, rather than relying on the default settings or manual guessing.

**Best parameters found:** `{'C': 0.1, 'solver': 'liblinear'}`

Tuned model results on the test set: Accuracy 0.9912, Precision 0.9861, Recall 1.0000, F1-Score 0.9930.

### Baseline vs Tuned comparison

| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 0.9737 | 0.9912 |
| F1-Score | 0.9790 | 0.9930 |

The tuned model outperformed the baseline on every metric. The confusion matrices (see confusion_matrix_comparison.png) show the tuned model made only 1 misclassification on the test set, compared to 3 for the baseline.

## What I learned about model evaluation

- Accuracy alone can be misleading, especially with an imbalanced dataset like this one. A model could reach high accuracy just by predicting the majority class (benign) most of the time, which is why Precision, Recall, and F1-Score matter alongside it.
- In a medical context like this dataset, Recall for the malignant class is especially important: a false negative (predicting benign when the tumor is actually malignant) is a far more costly mistake than a false positive. The confusion matrix makes this trade-off visible in a way a single accuracy number does not.
- Overfitting happens when a model performs well on training data but poorly on unseen test data, meaning it memorized patterns specific to the training set instead of learning something generalizable. Underfitting is the opposite: the model is too simple to capture the real pattern, and performs poorly on both training and test data. Comparing performance across cross-validation folds, rather than a single train-test split, gives a more honest picture of whether a model generalizes well.

## What hyperparameter tuning is and why it matters

Hyperparameters are settings chosen before training that control how a model learns, such as the regularization strength (C) in Logistic Regression - they are not learned from the data itself, unlike the model's actual coefficients. Choosing them manually is guesswork; GridSearchCV automates this by systematically trying every combination in a defined grid and using cross-validation to evaluate each one fairly, rather than judging based on a single train-test split that could be misleading by chance.

## Challenges faced

- Without feature scaling, Logistic Regression threw convergence warnings during GridSearchCV, since some features are on scales hundreds of times larger than others. Adding StandardScaler (fit only on the training data) resolved this and also made the baseline and tuned model comparison fair, since both were now evaluated on the same scaled features.
- Making sure the baseline model used the exact same train-test split and scaling as the tuned model, so the "before vs after" comparison reflects the actual effect of tuning, and not some other incidental difference between the two runs.

## Files

- dataset_exploration.py: loading and exploring the dataset
- baseline_model.py: baseline Logistic Regression model and evaluation
- hyperparameter_tuning.py: GridSearchCV tuning and baseline vs tuned comparison
- breast_cancer_prediction_pipeline.py: full mini project pipeline, including the confusion matrix heatmap
- confusion_matrix_comparison.png: side-by-side confusion matrix heatmap (baseline vs tuned)
- README.md: this file

## Author

Hifsa Iftikhar
