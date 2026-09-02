from fastapi import FastAPI

from app.routes.prediction import router as prediction_router
from app.services.detector import detector


app = FastAPI(
    title="YOLOv8n Car Detection API",
    description="FastAPI API for car detection using pretrained YOLOv8n",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": detector.model is not None,
        "model": detector.model_name
    }


app.include_router(prediction_router)