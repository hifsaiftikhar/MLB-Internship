# Student Grading System

A command-line Python program that collects student data, calculates subject averages, and assigns letter grades.

## What it does
- Takes input for one or more students: name, class, subjects, and marks
- Calculates each student's average across all entered subjects
- Assigns a letter grade based on the average
- Detects and skips duplicate subject names for the same student
- Prints a formatted report for each student
- Saves all results to results.json

## Grading criteria
| Average | Grade |
|---------|-------|
| 90 and above | A |
| 75 to 89 | B |
| 50 to 74 | C |
| Below 50 | F |

## How to run
Optional: create and activate a virtual environment first.
python -m venv venv
venv\Scripts\activate

Then run:
python grading_system.py
The program will ask:
1. How many students
2. For each student: name, class, number of subjects
3. For each subject: subject name and marks (0 to 100)

Output is printed to the terminal and saved in results.json.

## Input validation
- Student count, subject count, and marks are all validated
- Marks must be a number between 0 and 100
- Duplicate subject names are detected and skipped with a warning
- The program re-prompts on invalid input instead of crashing

## Design notes
The Student class stores each student's name, class, and subjects (as a dictionary of subject-to-marks). Average and grade calculation are methods on the class, so each student's data and logic stay self-contained.

Numeric input validation is handled by a standalone `get_valid_number()` function, shared across student count, subject count, and marks entry, so the same validation logic isn't duplicated three times.

## Files
- grading_system.py: main program
- results.json: sample output from a test run
- .gitignore: excludes venv and cache files from version control

## Author
Hifsa Iftikhar

## Future Improvements
- Add unit tests for average and grade calculation