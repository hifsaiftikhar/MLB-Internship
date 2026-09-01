from fastapi import FastAPI
from app.routes.students import router as students_router

app = FastAPI(
    title="Student Management API",
    description="A simple REST API to manage student records. Built with FastAPI.",
    version="1.0.0"
)

# Include routers
app.include_router(students_router)

@app.get("/", tags=["General"])
def root():
    """Welcome message."""
    return {"message": "Welcome to the Student Management API", "docs": "/docs"}

@app.get("/health", tags=["General"])
def health_check():
    """API health check."""
    return {"status": "ok", "message": "API is running"}