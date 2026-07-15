# Day-6: Data Preprocessing and Linear Regression

## What I worked on

Today's focus was preparing data for Machine Learning and training a first model using Scikit-learn: encoding categorical data, feature scaling, train-test splitting, and building a Linear Regression model to predict a student's Average_Score.

## Data preprocessing (data_preprocessing.py)

- Loaded the cleaned dataset from Day-5 (cleaned_student_performance.csv), which already includes the Average_Score column.
- Encoded the categorical Performance column using Label Encoding, to demonstrate the technique.
- Selected Math, Python, and Statistics as feature columns (X), and Average_Score as the target (y).
- Split the data into 80% training and 20% testing using train_test_split.
- Applied feature scaling (StandardScaler), fitting the scaler only on the training data and then using it to transform both the training and test sets.

### What I learned about data preprocessing

- Features (X) are the input columns a model learns from, and the target (y) is the value it's trying to predict.
- Categorical columns need to be converted into numbers before a model can use them, since algorithms like Linear Regression only work with numeric input. Label Encoding assigns each category a number, which works for this dataset but can imply an order between categories that doesn't really exist, so One-Hot Encoding is often preferred when the categories have no natural ranking.
- Feature scaling matters because features on very different scales (in this case, all four subjects happen to be on the same 0-100 scale, but this isn't always true) can cause some features to dominate a model simply due to their scale, not their actual importance.
- Data leakage is a real risk, not just a theoretical concern. In this dataset, the Performance column is directly calculated from Average_Score, which is the target I'm trying to predict. Including Performance as a feature would let the model "cheat" by indirectly seeing the answer, so it was excluded from the feature set entirely, and only encoded to demonstrate the technique.
- The scaler must be fit only on the training data, not the full dataset. Fitting it on everything (including test data) would leak information about the test set's distribution into training, which is another form of data leakage.

## Why train-test splitting is important

Splitting the data into training and testing sets lets you evaluate a model on data it has never seen before. If a model is only ever tested on the same data it was trained on, it can appear to perform very well simply by memorizing the training data, rather than actually learning a generalizable pattern. The 20% test set here acts as a small, honest check on how the model performs on new students it wasn't trained on.

## Linear Regression model (linear_regression_model.py)

A Linear Regression model was trained to predict Average_Score using Math, Python, and Statistics as features. Machine_Learning marks were deliberately left out of the features, so the model actually has to learn the relationship between the three included subjects and the overall average, rather than simply recomputing an exact mean from all four subjects (which would be a trivial, not very meaningful prediction).

### Evaluation metrics used

- Mean Absolute Error (MAE): the average absolute difference between actual and predicted scores.
- Mean Squared Error (MSE): similar to MAE, but squares the errors first, which penalizes larger mistakes more heavily.
- R-squared (R2) Score: how much of the variation in Average_Score is explained by the model, on a scale up to 1.0.

### Model performance and observations

On the test set:
- MAE: approximately 0.87
- MSE: approximately 0.82
- R2 Score: approximately 0.9975

The model performs very well on this dataset, which makes sense: Average_Score is closely related to the three subjects used as features (Math, Python, Statistics), even without Machine_Learning marks included. The small errors that do appear come from the missing ML component, which the model can only approximate rather than know exactly.

One honest limitation: the dataset only has 16 students, so the test set is just 4 rows. A result this clean on such a small test set should be read cautiously - it demonstrates the workflow correctly, but a real-world model would need a much larger dataset before this level of performance could be trusted.

## Mini Project: Student Score Prediction System (student_score_prediction_system.py)

Combines the full workflow into one script:
- Loads and preprocesses the dataset
- Trains the Linear Regression model
- Predicts Average_Score on the test set
- Displays MAE, MSE, and R2 Score
- Prints a comparison table of Actual vs Predicted scores
- Generates a scatter plot of Actual vs Predicted values, with a reference line showing what a perfect prediction would look like, saved as actual_vs_predicted.png

## Files

- cleaned_student_performance.csv: dataset carried over from Day-5
- data_preprocessing.py: preprocessing steps (encoding, feature/target selection, train-test split, scaling)
- linear_regression_model.py: model training and evaluation
- student_score_prediction_system.py: full mini project, including the scatter plot
- actual_vs_predicted.png: scatter plot of actual vs predicted scores
- README.md: this file

## Author

Hifsa Iftikhar
