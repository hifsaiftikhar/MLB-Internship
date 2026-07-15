import pandas as pd

# 1. Load the raw dataset
df = pd.read_csv("student_performance.csv")

print("Original dataset shape:", df.shape)

# 2. Check for missing values
print("\nMissing values per column before cleaning:")
print(df.isnull().sum())

# Fill missing marks with the average of that subject
subjects = ["Math", "Python", "Machine_Learning", "Statistics"]
for subject in subjects:
    df[subject] = df[subject].fillna(df[subject].mean())

print("\nMissing values per column after filling:")
print(df.isnull().sum())

# 3. Remove duplicate records
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicate_count}")
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# 4. Rename a column (Machine_Learning -> ML for a shorter, cleaner name)
df = df.rename(columns={"Machine_Learning": "ML"})
subjects = ["Math", "Python", "ML", "Statistics"]  # update subject list to match new name

# 5. Create a new column: Average_Score
df["Average_Score"] = df[subjects].mean(axis=1)

# 6. Create a new column: Performance category, based on Average_Score
def get_performance(score):
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Average"
    else:
        return "Needs Improvement"

df["Performance"] = df["Average_Score"].apply(get_performance)

# 7. Sort by Average_Score, highest first
df = df.sort_values(by="Average_Score", ascending=False)

print("\nCleaned dataset (sorted by Average_Score):")
print(df)

# 8. Save the cleaned dataset
df.to_csv("cleaned_student_performance.csv", index=False)
print("\nCleaned dataset saved as cleaned_student_performance.csv")
