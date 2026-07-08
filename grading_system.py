import json

class Student:
    def __init__(self, name, class_name):
        self.name = name
        self.class_name = class_name
        self.subjects = {}

    def add_subject(self, subject_name, marks):
        self.subjects[subject_name] = marks

    def calculate_average(self):
        count = len(self.subjects)
        if count == 0:
            return 0
        total = sum(self.subjects.values())
        return total / count

    def get_grade(self, average):
        if average >= 90:
            return "A"
        elif average >= 75:
            return "B"
        elif average >= 50:
            return "C"
        else:
            return "F"


students = []

while True:
    try:
        num_students = int(input("How many students? "))
        if num_students > 0:
            break
        else:
            print("Number of students must be greater than 0.")
    except ValueError:
        print("Please enter a valid whole number.")

for i in range(num_students):
    name = input("Enter student name: ").strip()
    class_name = input("Enter class: ").strip()
    student = Student(name, class_name)

    while True:
        try:
            num_subjects = int(input(f"How many subjects for {name}? "))
            if num_subjects > 0:
                break
            else:
                print("Number of subjects must be greater than 0.")
        except ValueError:
            print("Please enter a valid whole number.")

    for j in range(num_subjects):
        subject_name = input("Enter subject name: ").strip()
        while True:
            try:
                marks = float(input(f"Enter marks for {subject_name}: "))
                if 0 <= marks <= 100:
                    break
                else:
                    print("Marks must be between 0 and 100. Try again.")
            except ValueError:
                print("Please enter a valid number.")
        student.add_subject(subject_name, marks)

    students.append(student)

for s in students:
    print(f"\nName: {s.name}")
    print(f"Class: {s.class_name}")
    print("Subjects and Marks:")
    for subject, marks in s.subjects.items():
        print(f"  {subject}: {marks}")
    avg = s.calculate_average()
    grade = s.get_grade(avg)
    print(f"Average: {avg:.2f}")
    print(f"Grade: {grade}")

results = []

for s in students:
    avg = s.calculate_average()
    grade = s.get_grade(avg)
    student_data = {
        "name": s.name,
        "class": s.class_name,
        "subjects": s.subjects,
        "average": avg,
        "grade": grade,
    }
    results.append(student_data)

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results.json")