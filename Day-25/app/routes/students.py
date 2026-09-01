from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.student import StudentCreate, StudentUpdate, StudentResponse

router = APIRouter(prefix="/students", tags=["Students"])

# In-memory storage
students_db: dict = {}
counter: int = 1

@router.get("/", response_model=List[StudentResponse])
def get_all_students():
    """Return all students."""
    return list(students_db.values())

@router.post("/", response_model=StudentResponse, status_code=201)
def add_student(student: StudentCreate):
    """Add a new student."""
    global counter
    new_student = {"id": counter, **student.dict()}
    students_db[counter] = new_student
    counter += 1
    return new_student

@router.get("/search", response_model=List[StudentResponse])
def search_student(name: str):
    """Search students by name."""
    results = [s for s in students_db.values() if name.lower() in s["name"].lower()]
    if not results:
        raise HTTPException(status_code=404, detail=f"No students found with name '{name}'")
    return results

@router.get("/{student_id}", response_model=StudentResponse)
def get_student(student_id: int):
    """Get a specific student by ID."""
    if student_id not in students_db:
        raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
    return students_db[student_id]

@router.put("/{student_id}", response_model=StudentResponse)
def update_student(student_id: int, updated: StudentUpdate):
    """Update student information."""
    if student_id not in students_db:
        raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
    student = students_db[student_id]
    update_data = updated.dict(exclude_unset=True)
    student.update(update_data)
    students_db[student_id] = student
    return student

@router.delete("/{student_id}")
def delete_student(student_id: int):
    """Delete a student."""
    if student_id not in students_db:
        raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
    del students_db[student_id]
    return {"message": f"Student with ID {student_id} deleted successfully"}