# YOLOv8n Car Detection API

A FastAPI-based REST API for detecting cars in uploaded images using the pretrained YOLOv8n object detection model.

## Project Overview

This project provides an API that accepts an image, runs YOLOv8n inference, and returns car detection results.

The API provides:

- Health check endpoint
- Car detection with JSON response
- Bounding box coordinates
- Confidence scores
- Configurable confidence threshold
- Processed image with bounding boxes
- Image validation and error handling
- Swagger UI for API testing

## Tech Stack

- Python
- FastAPI
- Ultralytics YOLOv8n
- OpenCV
- Pillow
- Pydantic
- Uvicorn

## Project Structure

```text
Day-26/
│
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── prediction.py
│   ├── services/
│   │   └── detector.py
│   └── schemas/
│       └── response.py
│
├── models/
│   └── yolov8n.pt
│
├── requirements.txt
└── README.md
````

## Model

The project uses the pretrained YOLOv8n model from Ultralytics.

The model is used for detecting the `car` class from the COCO-trained classes.

Model file:

```text
models/yolov8n.pt
```

No additional model training is required for this API.

## Installation

Clone the repository and open the project directory:

```bash
cd Day-26
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

Start the FastAPI server using:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

The available endpoints are:

* `GET /health`
* `POST /predict`
* `POST /predict/image`

## API Endpoints

### 1. Health Check

**Endpoint:**

```text
GET /health
```

Checks whether the API is running and the YOLO model is loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "yolov8n.pt"
}
```

### 2. Car Prediction

**Endpoint:**

```text
POST /predict
```

Accepts an image and returns detected cars as JSON.

### Parameters

| Parameter  | Type       | Description                  |
| ---------- | ---------- | ---------------------------- |
| file       | Image file | Input image                  |
| confidence | Float      | Minimum confidence threshold |

The confidence threshold must be between `0.0` and `1.0`.

Default:

```text
0.25
```

### Example Response

```json
{
  "detections": [
    {
      "class_name": "car",
      "confidence": 0.9001,
      "bbox": [
        21,
        111,
        540,
        287
      ]
    }
  ],
  "total": 1,
  "model": "yolov8n.pt",
  "confidence_threshold": 0.25
}
```

### Response Fields

* `class_name` - Detected object class
* `confidence` - Detection confidence score
* `bbox` - Bounding box coordinates in `[x1, y1, x2, y2]` format
* `total` - Total number of detected cars
* `model` - Model used for inference
* `confidence_threshold` - Confidence threshold used for the request

### 3. Processed Image

**Endpoint:**

```text
POST /predict/image
```

Accepts an image and returns the processed image with detection bounding boxes.

Example:

```text
POST /predict/image?confidence=0.50
```

The response is returned as:

```text
image/jpeg
```

## Confidence Threshold

The confidence threshold controls which detections are returned.

Examples:

```text
confidence=0.25
confidence=0.50
confidence=0.75
```

A higher threshold generally results in fewer detections because only detections with higher confidence are kept.

## Validation and Error Handling

The API validates uploaded files before running inference.

Supported image types:

* JPG / JPEG
* PNG
* WEBP

The API handles:

* Unsupported file types
* Empty files
* Invalid image files
* Invalid confidence values

Example unsupported file response:

```json
{
  "detail": "Unsupported file type. Please upload JPG, PNG, or WEBP."
}
```

## Testing

The API was tested using FastAPI Swagger UI.

Test cases include:

1. Health check
2. Valid car image
3. Multiple car detections
4. Confidence threshold `0.25`
5. Confidence threshold `0.50`
6. Confidence threshold `0.75`
7. Processed image with bounding boxes
8. Unsupported file type
9. Empty file
10. Invalid image
11. Image with no detected cars

## Inference Flow

```text
Client
   ↓
FastAPI
   ↓
Image Validation
   ↓
YOLOv8n Model
   ↓
Car Detection
   ↓
Inference Results
   ↓
JSON Response / Processed Image
```

## API Testing Tools

The API can be tested using:

* Swagger UI
* Postman
* Thunder Client

## Requirements

All required Python packages are listed in:

```text
requirements.txt
```

## Author

Hifsa Iftikhar