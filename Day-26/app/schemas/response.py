from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]

class PredictionResponse(BaseModel):
    detections: List[Detection]
    total: int
    model: str
    confidence_threshold: float