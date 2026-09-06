from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routes.video import router as video_router
from app.utils.logger import logger

APP_VERSION = "2.0.0"

app = FastAPI(
    title="Production-Ready AI Video Processing API",
    description="Upload a video, process it with YOLO object detection, track job status, and download the annotated result.",
    version=APP_VERSION
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} | Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred. Please try again.",
            "detail": str(exc)
        }
    )

app.include_router(video_router)

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the AI Video Processing API",
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "upload_video": "POST /video/process",
            "check_status": "GET /video/status/{job_id}",
            "download_result": "GET /video/result/{job_id}"
        }
    }

@app.get("/health", tags=["General"])
def health():
    from app.services.detector import detector
    logger.info("Health check requested.")
    return {
        "success": True,
        "status": "ok",
        "version": APP_VERSION,
        "model": {
            "loaded": detector.model is not None,
            "name": detector.model_name
        }
    }