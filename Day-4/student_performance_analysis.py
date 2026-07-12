import pandas as pd

# 1. Load the dataset
df = pd.read_csv("students.csv")

# 2. Display basic information about the dataset
print("Dataset info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

# 3. Calculate average marks for each subject
subjects = ["Math", "Science", "English"]
print("\nAverage marks per subject:")
print(df[subjects].mean())

# Also calculate each student's overall average, to use for the rest of the analysis
df["Average"] = df[subjects].mean(axis=1)

# 4. Identify the top 5 performing students (based on overall average)
top_5 = df.sort_values(by="Average", ascending=False).head(5)
print("\nTop 5 performing students:")
print(top_5[["Name", "Roll", "Average"]])

# 5. Find students scoring below the overall class average
overall_average = df["Average"].mean()
below_average = df[df["Average"] < overall_average]
print(f"\nOverall class average: {overall_average:.2f}")
print("Students scoring below the average:")
print(below_average[["Name", "Roll", "Average"]])

# 6. Display the total number of students
print("\nTotal number of students:", len(df))

# 7. Save the analyzed dataset as a new CSV file
df.to_csv("analyzed_students.csv", index=False)
print("\nAnalyzed dataset saved as analyzed_students.csv")
