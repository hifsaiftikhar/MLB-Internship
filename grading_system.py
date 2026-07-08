import json


def get_valid_number(prompt, min_val=None, max_val=None, is_float=True):
    while True:
        try:
            value = float(input(prompt)) if is_float else int(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Must be at least {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Must be at most {max_val}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


class Student:
    def __init__(self, name, class_name):
        self.name = name
        self.class_name = class_name
        self.student_id = (name, class_name)  # tuple: fixed identity, never changes
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

num_students = int(get_valid_number("How many students? ", min_val=1, is_float=False))

for i in range(num_students):
    name = input("Enter student name: ").strip()
    class_name = input("Enter class: ").strip()
    student = Student(name, class_name)

    num_subjects = int(get_valid_number(f"How many subjects for {name}? ", min_val=1, is_float=False))

    entered_subjects = set()  # tracks subject names already entered, prevents silent duplicates

    for j in range(num_subjects):
        subject_name = input("Enter subject name: ").strip()
        if subject_name in entered_subjects:
            print(f"{subject_name} already entered for this student. Skipping duplicate.")
            continue
        entered_subjects.add(subject_name)

        marks = get_valid_number(f"Enter marks for {subject_name}: ", min_val=0, max_val=100)
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