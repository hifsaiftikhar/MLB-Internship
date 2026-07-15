import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cleaned_student_performance.csv")

subjects = ["Math", "Python", "ML", "Statistics"]

print("===== STUDENT PERFORMANCE DASHBOARD =====\n")

# 1. How many students are in the dataset?
total_students = len(df)
print(f"Total number of students: {total_students}\n")

# 2. What is the average score for each subject?
subject_averages = df[subjects].mean()
print("Average score for each subject:")
print(subject_averages)
print()

# 3. Who are the Top 5 performing students?
top_5 = df.sort_values(by="Average_Score", ascending=False).head(5)
print("Top 5 performing students:")
print(top_5[["Name", "Average_Score"]])
print()

# 4. Which students need improvement?
needs_improvement = df[df["Performance"] == "Needs Improvement"]
print("Students who need improvement:")
print(needs_improvement[["Name", "Average_Score"]])
print()

# 5. Which subject has the highest class average?
top_subject = subject_averages.idxmax()
print(f"Subject with the highest class average: {top_subject} ({subject_averages.max():.2f})\n")

# 6. Visualize the findings

# Top 5 students - bar chart
plt.figure(figsize=(8, 5))
plt.bar(top_5["Name"], top_5["Average_Score"], color="mediumpurple")
plt.title("Top 5 Performing Students")
plt.xlabel("Student")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("dashboard_top5_students.png")
plt.close()

# Subject averages - bar chart
plt.figure(figsize=(8, 5))
plt.bar(subject_averages.index, subject_averages.values, color="teal")
plt.title("Average Score per Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("dashboard_subject_averages.png")
plt.close()

print("Dashboard charts saved: dashboard_top5_students.png, dashboard_subject_averages.png")
