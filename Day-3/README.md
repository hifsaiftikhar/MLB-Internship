# Day-3: File Handling, JSON, and Persistent Student Record System

## What I worked on

Today's focus was learning how to work with files and structured data in Python — reading and writing text files, and storing structured data using JSON. I then upgraded my Student Record Management System from Day-2 so that student records are saved permanently to a JSON file instead of being lost when the program closes.

## Practice programs

### File Handling (file_practice.py)
- Created a text file and wrote data into it
- Read and displayed the file's contents
- Appended new data to the existing file without erasing what was already there
- Counted the number of lines in the file using readlines()

### JSON (json_practice.py)
- Stored student information in a JSON file using json.dump()
- Read the data back using json.load()
- Updated an existing student's information and saved the change
- Restructured the data to hold multiple students and added a new one

## Student Record Management System (persistent version)

Upgraded the Day-2 CRUD system (Add, View, Search, Update, Delete) with two new functions:

- load_students(): reads students.json when the program starts. If the file does not exist yet (first run), it returns an empty dictionary instead of crashing, using a try/except around FileNotFoundError.
- save_students(): writes the current student dictionary to students.json using json.dump().

save_students() is called at the end of add_student, update_student, and delete_student, so every change is written to disk immediately rather than only when the program exits. This means the data survives even if the program is closed unexpectedly.

### How file handling and JSON work together

File handling is the general mechanism for reading and writing to disk — opening a file, writing text, closing it. JSON is a structured text format that fits naturally into this: instead of writing plain, unstructured text, json.dump() writes a Python dictionary to a file in a structured, readable format, and json.load() reads it back into a real Python dictionary. In this project, the file handling piece is the with open(...) block, and the JSON piece is what actually converts between the students dictionary in memory and the text saved on disk.

### Exception handling

- Age input is validated using try/except around ValueError, so entering a letter does not crash the program.
- Reading the JSON file at startup is wrapped in try/except around FileNotFoundError, so a first-time run (with no existing file) starts with an empty student list instead of crashing.
- Name and course fields are checked to ensure they are not left blank.

## Challenges faced

- Initially had file path issues where students.json was created in the wrong folder because the program was run from a different working directory than expected. Resolved by always running the script from inside the Day-3 folder directly.
- Had to restructure the JSON practice data from a single student dictionary into a dictionary of dictionaries (keyed by roll number) to support adding multiple students, consistent with the design used in the main project.

## Files

- file_practice.py: file handling practice problems
- json_practice.py: JSON practice problems
- sample.txt: sample output from file handling practice
- student.json: sample output from JSON practice
- student_record_system.py: persistent Student Record Management System
- students.json: sample saved student data from the CRUD system
- README.md: this file

## Author

Hifsa Iftikhar
