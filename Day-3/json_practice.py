#Practice Problem 1: Store student information in a JSON file.
import json

student = {"name": "Ali", "roll": "101", "age": 20, "course": "AI"}

with open("student.json", "w") as f:
    json.dump(student, f, indent=4)

#Practice Problem 2: Read data from a JSON file.
import json

with open("student.json", "r") as f:
    data = json.load(f)
    print(data)

#Practice Problem 3: Update an existing student's information (in the JSON file).
with open("student.json", "r") as f:
    data = json.load(f)

data["course"] = "Data Science"

with open("student.json", "w") as f:
    json.dump(data, f, indent=4)

#Practice Problem 4: Add a new student to the JSON file.
with open("student.json", "r") as f:
    data = json.load(f)

# wrap the single student under their roll number
students = {data["roll"]: {"name": data["name"], "age": data["age"], "course": data["course"]}}

with open("student.json", "w") as f:
    json.dump(students, f, indent=4)

# load the existing students, add a new one, save back:
with open("student.json", "r") as f:
    students = json.load(f)

students["102"] = {"name": "Sara", "age": 21, "course": "AI"}

with open("student.json", "w") as f:
    json.dump(students, f, indent=4)