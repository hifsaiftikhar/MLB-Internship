import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset (produced by data_cleaning.py)
df = pd.read_csv("cleaned_student_performance.csv")

subjects = ["Math", "Python", "ML", "Statistics"]

# 1. Bar chart: average marks of each student
plt.figure(figsize=(10, 6))
plt.bar(df["Name"], df["Average_Score"], color="steelblue")
plt.title("Average Marks per Student")
plt.xlabel("Student")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart_average_marks.png")
plt.close()

# 2. Histogram: distribution of Average Scores
plt.figure(figsize=(8, 6))
plt.hist(df["Average_Score"], bins=6, color="mediumseagreen", edgecolor="black")
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("histogram_average_score.png")
plt.close()

# 3. Scatter plot: Python vs Machine Learning marks
plt.figure(figsize=(8, 6))
plt.scatter(df["Python"], df["ML"], color="darkorange")
plt.title("Python Marks vs Machine Learning Marks")
plt.xlabel("Python")
plt.ylabel("Machine Learning")
plt.tight_layout()
plt.savefig("scatter_python_ml.png")
plt.close()

# 4. Pie chart: distribution of students by Performance category
performance_counts = df["Performance"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(performance_counts, labels=performance_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Student Distribution by Performance Category")
plt.tight_layout()
plt.savefig("pie_chart_performance.png")
plt.close()

# 5. Box plot: spread of marks across all subjects
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[subjects])
plt.title("Spread of Marks Across Subjects")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("box_plot_subjects.png")
plt.close()

print("All 5 charts generated and saved as PNG files.")
