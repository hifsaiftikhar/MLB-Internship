def add_student(students):
    roll = input("Enter roll number: ").strip()
    
    if roll in students:
        print("A student with this roll number already exists.")
    else:
        while True:
            name = input("Enter name: ").strip()
            if name:
                break
            else:
                print("Name cannot be blank.")
        
        while True:
            try:
                age = int(input("Enter age: "))
                if age > 0:
                    break
                else:
                    print("Age must be a positive number.")
            except ValueError:
                print("Please enter a valid whole number for age.")
        
        while True:
            course = input("Enter course: ").strip()
            if course:
                break
            else:
                print("Course cannot be blank.")
        
        students[roll] = {"name": name, "age": age, "course": course}
        print(f"Student {name} added successfully.")


def view_students(students):
    if not students:
        print("No students in the system.")
        return
    
    for roll, details in students.items():
        print(f"Roll: {roll}, Name: {details['name']}, Age: {details['age']}, Course: {details['course']}")


def search_student(students):
    roll = input("Enter roll number to search: ").strip()
    
    if roll in students:
        details = students[roll]
        print(f"Roll: {roll}, Name: {details['name']}, Age: {details['age']}, Course: {details['course']}")
    else:
        print("Student not found.")


def update_student(students):
    roll = input("Enter roll number to update: ").strip()
    
    if roll not in students:
        print("Student not found.")
        return
    
    print("What do you want to update? 1. Name 2. Age 3. Course")
    choice = input("Enter choice (1/2/3): ")
    
    if choice == "1":
        while True:
            new_name = input("Enter new name: ").strip()
            if new_name:
                break
            else:
                print("Name cannot be blank.")
        students[roll]["name"] = new_name
    elif choice == "2":
        while True:
            try:
                new_age = int(input("Enter new age: "))
                if new_age > 0:
                    break
                else:
                    print("Age must be a positive number.")
            except ValueError:
                print("Please enter a valid whole number for age.")
        students[roll]["age"] = new_age
    elif choice == "3":
        while True:
            new_course = input("Enter new course: ").strip()
            if new_course:
                break
            else:
                print("Course cannot be blank.")
        students[roll]["course"] = new_course
    else:
        print("Invalid choice.")
        return
    
    print("Student updated successfully.")


def delete_student(students):
    roll = input("Enter roll number to delete: ").strip()
    
    if roll in students:
        del students[roll]
        print("Student deleted successfully.")
    else:
        print("Student not found.")


def main():
    students = {}
    
    while True:
        print("\n--- Student Record Management System ---")
        print("1. Add Student")
        print("2. View All Students")
        print("3. Search Student")
        print("4. Update Student")
        print("5. Delete Student")
        print("6. Total Number of Students")
        print("7. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            add_student(students)
        elif choice == "2":
            view_students(students)
        elif choice == "3":
            search_student(students)
        elif choice == "4":
            update_student(students)
        elif choice == "5":
            delete_student(students)
        elif choice == "6":
            print(f"Total students: {len(students)}")
        elif choice == "7":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")


main()