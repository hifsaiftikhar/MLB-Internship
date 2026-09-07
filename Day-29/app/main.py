from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routes.video import router as video_router, jobs_router
from app.database.database import create_tables
from app.utils.logger import logger

APP_VERSION = "3.0.0"

app = FastAPI(
    title="AI Video Processing API with Database",
    description="Upload videos, process with YOLO, track jobs permanently in SQLite database.",
    version=APP_VERSION
)

# Create database tables on startup
@app.on_event("startup")
def startup():
    logger.info("Starting API - creating database tables...")
    create_tables()
    logger.info("Database tables ready.")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} | Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred.",
            "detail": str(exc)
        }
    )

app.include_router(video_router)
app.include_router(jobs_router)

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the AI Video Processing API",
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "upload_video": "POST /video/process",
            "get_all_jobs": "GET /jobs",
            "get_job": "GET /jobs/{job_id}",
            "delete_job": "DELETE /jobs/{job_id}",
            "download_result": "GET /video/result/{job_id}"
        }
    }

@app.get("/health", tags=["General"])
def health():
    from app.services.detector import detector
    return {
        "success": True,
        "status": "ok",
        "version": APP_VERSION,
        "model": {
            "loaded": detector.model is not None,
            "name": detector.model_name
        }
    }