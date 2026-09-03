from fastapi import FastAPI
from app.routes.video import router as video_router

app = FastAPI(
    title="AI Video Processing API",
    description="Upload a video, process it with YOLO object detection, and download the annotated result.",
    version="1.0.0"
)

app.include_router(video_router)

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the AI Video Processing API",
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
    return {
        "status": "ok",
        "model_loaded": detector.model is not None,
        "model": detector.model_name
    }