from pydantic import BaseModel, Field
from typing import Optional

class StudentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=5, le=100)
    course: str = Field(..., min_length=1, max_length=100)
    grade: Optional[str] = None

class StudentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=5, le=100)
    course: Optional[str] = Field(None, min_length=1, max_length=100)
    grade: Optional[str] = None

class StudentResponse(BaseModel):
    id: int
    name: str
    age: int
    course: str
    grade: Optional[str] = None