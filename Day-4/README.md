# Day-4: NumPy, Pandas, and Student Performance Analysis

## What I worked on

Today's focus was moving into Python for Data Science, starting with the two core libraries used throughout AI and ML work: NumPy for numerical arrays, and Pandas for tabular data. I practiced both separately, then used Pandas to build a small student performance analysis project.

## NumPy practice (numpy_practice.py)

- Created 1D and 2D arrays using np.array()
- Performed arithmetic operations on arrays (addition, multiplication), which apply element-wise across the whole array at once
- Found max, min, mean, and sum of an array using built-in array methods
- Reshaped a 1D array of 12 numbers into 3x4 and 4x3 grids using .reshape()
- Practiced indexing and slicing on both 1D and 2D arrays, including selecting a specific row or column from a 2D array

### What I learned about NumPy

NumPy arrays behave differently from plain Python lists: operations like `arr + 5` or `arr * 2` apply to every element at once, without needing a loop. This is both faster and more readable than manually looping through a list. Reshaping is useful because the same data can be viewed in different dimensional structures depending on what an operation needs.

## Pandas practice (pandas_practice.py)

- Loaded the sample dataset (students.csv) using pd.read_csv()
- Displayed the first and last five rows using .head() and .tail()
- Displayed dataset structure and data types using .info()
- Found missing values per column using .isnull().sum()
- Filtered rows based on a condition (students scoring above 80 in Math)
- Displayed summary statistics using .describe()

### What I learned about Pandas

A Pandas DataFrame is essentially a table, similar to a spreadsheet, where each column can hold a different data type. .info() and .describe() give a fast overview of a dataset's shape, types, and distribution before doing any real analysis. Filtering with a condition like df[df["Math"] > 80] returns only the rows that satisfy it, which is the basis for most data exploration tasks.

## Mini Project: Student Performance Analysis (student_performance_analysis.py)

Using the same student dataset, this project:

- Loads the dataset and displays basic info
- Calculates the average marks for each subject (Math, Science, English)
- Calculates each student's overall average across subjects
- Identifies the top 5 performing students by overall average
- Finds students scoring below the overall class average
- Displays the total number of students
- Saves the analyzed dataset (with the added Average column) as analyzed_students.csv

### Key insights from the dataset

- The class average across all three subjects came out to about 72, with Math having the widest spread between the highest and lowest scores.
- The top-performing student (Ayesha) scored consistently high across all three subjects rather than excelling in just one.
- Almost half the class (7 out of 15 students) scored below the overall average, which is expected in a small dataset with a few very high and very low scorers pulling the average.
- Two students had missing marks (one missing a Math score, one missing a Science score), which had to be accounted for when calculating averages, since Pandas automatically excludes missing values from mean calculations rather than treating them as zero.

## Challenges faced

- Handling missing values correctly: a missing mark should not be treated as a zero when calculating an average, since that would unfairly lower a student's overall average. Pandas handles this automatically by excluding NaN values from .mean(), which is different from how a manual sum-and-divide calculation would behave.
- Deciding how to rank "top performers" required calculating an overall average column first, rather than ranking by a single subject, so that a student's overall performance was reflected fairly.

## Files

- numpy_practice.py: NumPy practice programs
- pandas_practice.py: Pandas practice programs
- student_performance_analysis.py: mini project analyzing student performance
- students.csv: original dataset
- analyzed_students.csv: processed dataset with calculated averages
- README.md: this file

## Author

Hifsa Iftikhar
