# Day-25: Student Management REST API

## What is a REST API?
A REST API is a way for applications to communicate over the internet using standard HTTP methods. Instead of a user clicking buttons on a website, another program sends requests to specific URLs (endpoints) and gets back data — usually in JSON format.

REST stands for Representational State Transfer. It follows simple rules:
- Each endpoint represents a resource (like students)
- HTTP methods define the action (GET to read, POST to create, PUT to update, DELETE to remove)
- Responses are stateless — each request contains everything needed, no session memory

## GET vs POST
| Method | Purpose | Sends Data? | Example |
|--------|---------|-------------|---------|
| GET | Read data | No (only in URL) | Get all students |
| POST | Create data | Yes (in request body) | Add a new student |
| PUT | Update data | Yes (in request body) | Update student info |
| DELETE | Remove data | No | Delete a student |

## What is Pydantic?
Pydantic is a Python library that validates data automatically. When a request comes in, Pydantic checks:
- Are all required fields present?
- Are the data types correct (string, int, etc.)?
- Do values meet constraints (min length, max value, etc.)?

If validation fails, FastAPI automatically returns a 422 error with a clear message — no manual checking needed.

## Project Structure
```
Day-25/
├── app/
│   ├── main.py          — FastAPI app, root and health endpoints
│   ├── routes/
│   │   └── students.py  — All student CRUD endpoints
│   └── schemas/
│       └── student.py   — Pydantic models for request/response validation
├── requirements.txt
└── README.md
```

## How to Run

**Install dependencies:**
```
pip install -r requirements.txt
```

**Start the server:**
```
uvicorn app.main:app --reload
```

**Open Swagger UI:**
```
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Welcome message |
| GET | /health | API health check |
| GET | /students/ | Get all students |
| POST | /students/ | Add a new student |
| GET | /students/{id} | Get student by ID |
| GET | /students/search?name= | Search by name |
| PUT | /students/{id} | Update student |
| DELETE | /students/{id} | Delete student |

## Example Request and Response

**POST /students/**
```json
Request body:
{
  "name": "Hifsa",
  "age": 21,
  "course": "Artificial Intelligence",
  "grade": "A"
}

Response:
{
  "id": 1,
  "name": "Hifsa",
  "age": 21,
  "course": "Artificial Intelligence",
  "grade": "A"
}
```

**GET /students/1**
```json
Response:
{
  "id": 1,
  "name": "Hifsa",
  "age": 21,
  "course": "Artificial Intelligence",
  "grade": "A"
}
```

**Error Response (student not found):**
```json
{
  "detail": "Student with ID 5 not found"
}
```

## Challenges Faced
- Understanding how routers work in FastAPI — keeping routes in a separate file and registering them in main.py
- Pydantic's `exclude_unset=True` for partial updates — only updating fields that were actually sent in the request

## Author
Hifsa Iftikhar