# Day-5: Data Cleaning, Visualization, and Student Performance Dashboard

## What I worked on

Today's focus was cleaning data with Pandas and visualizing it with Matplotlib and Seaborn, before building a small Student Performance Dashboard that answers specific questions about the dataset.

## Data cleaning (data_cleaning.py)

Steps performed on student_performance.csv:

- Checked for missing values using .isnull().sum(), which found one missing Python mark.
- Filled the missing value with the average of that subject, instead of dropping the row entirely, so one missing mark doesn't remove a student's whole record from the analysis.
- Removed duplicate records using .drop_duplicates() — one exact duplicate row was found and removed.
- Renamed the Machine_Learning column to ML for a shorter, cleaner name.
- Created a new column, Average_Score, calculated as the mean of all four subject columns for each student.
- Created a new column, Performance, categorizing each student as Excellent, Good, Average, or Needs Improvement based on their Average_Score.
- Sorted the dataset by Average_Score, highest first.
- Saved the result as cleaned_student_performance.csv.

## Data visualization (data_visualization.py)

Five charts were created from the cleaned dataset:

- Bar chart — average marks of each student, to compare performance across the whole class at a glance.
- Histogram — distribution of Average Scores, to see how scores are spread across the class (whether most students cluster around a certain range or are spread out evenly).
- Scatter plot — Python marks vs Machine Learning marks, to check whether students who do well in Python also tend to do well in Machine Learning.
- Pie chart — proportion of students in each Performance category, to see the overall shape of class performance at a glance.
- Box plot — spread of marks across all four subjects together, to compare which subjects have wider or narrower score ranges, and to spot any outliers.

All five charts are saved as PNG files in this folder.

## Mini Project: Student Performance Dashboard (student_performance_dashboard.py)

Answers the following directly from the cleaned dataset:

- Total number of students
- Average score for each subject
- Top 5 performing students
- Students who need improvement
- The subject with the highest class average

Two additional charts are generated here: a bar chart of the top 5 students, and a bar chart comparing subject averages.

## Key insights from the dataset

1. Python had the highest class average (about 76.3) among all four subjects, while Statistics had the lowest (about 70.5) — suggesting students found Python concepts easier to grasp than Statistics in this dataset.
2. Nearly half the class (7 out of 16 students) fell into the Needs Improvement category, showing a fairly wide performance gap between the top and bottom of the class rather than most students clustering in the middle.
3. The top-performing students were consistently strong across all subjects rather than excelling in just one — for example, the highest scorer (Ayesha) scored above 94 in every subject, rather than having one standout subject pulling up an otherwise average performance.

## Challenges faced

- Deciding how to handle the missing value: dropping the row would have lost an otherwise complete student record, so filling it with the subject's average was a more reasonable approach for this small dataset.
- Making sure the box plot and scatter plot used the renamed ML column consistently, since renaming Machine_Learning to ML early in the cleaning script meant every later step needed to reference the new column name.

## Files

- student_performance.csv: original dataset (contains a duplicate row and a missing value, for cleaning practice)
- data_cleaning.py: data cleaning script
- cleaned_student_performance.csv: cleaned dataset
- data_visualization.py: script generating all 5 charts
- bar_chart_average_marks.png, histogram_average_score.png, scatter_python_ml.png, pie_chart_performance.png, box_plot_subjects.png: generated charts
- student_performance_dashboard.py: mini project answering dashboard questions
- dashboard_top5_students.png, dashboard_subject_averages.png: dashboard charts
- README.md: this file

## Author

Hifsa Iftikhar
