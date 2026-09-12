from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes.video import router as video_router, jobs_router
from app.routes.auth import router as auth_router

from app.database.database import create_tables
from app.utils.logger import logger

APP_VERSION = "3.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting API - creating database tables...")
    create_tables()
    logger.info("Database tables ready.")
    yield
    # Shutdown (if needed)
    logger.info("Shutting down API...")


app = FastAPI(
    title="AI Video Processing API with Authentication",
    description="Secure AI Video Processing API with JWT authentication, user roles, and SQLite database.",
    version=APP_VERSION,
    lifespan=lifespan
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc} | Path: {request.url.path}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred.",
            "detail": str(exc)
        }
    )


# Include routers
app.include_router(auth_router)
app.include_router(video_router)
app.include_router(jobs_router)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the Secure AI Video Processing API",
        "version": APP_VERSION,
        "docs": "/docs",
        "authentication": {
            "register": "POST /auth/register",
            "login": "POST /auth/login",
            "profile": "GET /auth/me"
        },
        "endpoints": {
            "upload_video": "POST /video/process",
            "get_jobs": "GET /jobs/",
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