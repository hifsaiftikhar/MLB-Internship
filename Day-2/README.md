# Day-2: Python Data Structures and Problem Solving

## What I worked on

Today's focus was strengthening core Python data structures — lists, tuples, sets, and dictionaries — through practice problems, followed by building a Student Record Management System using dictionaries.

## Practice problems

### Lists
- Find the largest number in a list (manual iteration, no built-in max())
- Find the second largest number in a list
- Remove duplicate values from a list
- Reverse a list without using built-in reverse()
- Find common elements between two lists

### Tuples
- Count occurrences of an element in a tuple
- Convert a tuple into a list and back

### Sets
- Find unique values from a list
- Perform union and intersection operations on two sets

### Dictionaries
- Create a nested student record dictionary
- Calculate average marks across students
- Count frequency of words in a sentence

All practice problems are in practice_problems.py.

## Student Record Management System

A console-based CRUD application for managing student records, built using a dictionary of dictionaries keyed by roll number.

### Features
- Add Student
- View All Students
- Search Student (by roll number)
- Update Student Information
- Delete Student
- Display Total Number of Students
- Menu-driven interface
- Input validation on roll number and age

### Design decision: dictionary of dictionaries vs list of dictionaries

I used a dictionary keyed by roll number instead of a list of dictionaries. Since roll numbers are unique identifiers, this makes search, update, and delete operations direct (students[roll]) instead of requiring a loop through every record. This also naturally supports the search-by-roll-number requirement without extra logic.

## What I learned

- The practical difference between mutable (list, dict, set) and immutable (tuple) structures, and when each is the right choice
- How set operations (union, intersection) map directly to real deduplication and comparison problems
- How dictionary key design affects the efficiency of an entire program — choosing roll number as a key made search, update, and delete simpler than they would be with a list
- The importance of validating input at the point of entry, since a single bad input (like a letter typed for age) can crash an entire program and lose all in-memory data

## Challenges faced

- Roll numbers in practice are not always purely numeric (for example, university roll numbers can include letters), so I had to change roll number handling from integer to string partway through
- Initially missed input validation on the age field in both Add Student and Update Student, which caused the program to crash on non-numeric input; fixed by wrapping these inputs in validation loops

## Files

- practice_problems.py: all list, tuple, set, and dictionary practice problems
- student_record_system.py: Student Record Management System
- README.md: this file
