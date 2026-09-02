from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from PIL import Image, UnidentifiedImageError
import io

from app.services.detector import detector

router = APIRouter()


ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    ),
):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload JPG, PNG, or WEBP."
        )

    # Read uploaded file
    data = await file.read()

    # Check empty file
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Validate actual image
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Run YOLO inference
    try:
        all_detections = detector.predict(data, confidence)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Keep only cars
    detections = [
        detection
        for detection in all_detections
        if detection["class_name"] == "car"
    ]

    return {
        "detections": detections,
        "total": len(detections),
        "model": detector.model_name,
        "confidence_threshold": confidence
    }


@router.post("/predict/image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    ),
):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload JPG, PNG, or WEBP."
        )

    # Read uploaded file
    data = await file.read()

    # Check empty file
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Validate actual image
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Run inference and return annotated image
    try:
        annotated_image = detector.predict_with_image(data, confidence)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    from fastapi.responses import Response

    return Response(
        content=annotated_image,
        media_type="image/jpeg"
    )