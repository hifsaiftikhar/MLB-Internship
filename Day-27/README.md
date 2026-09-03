# Day-27: AI Video Processing API

## How Video Processing Works
A video is a sequence of frames. This API reads each frame one by one, runs YOLO object detection on it, draws bounding boxes with class names and confidence scores, then writes the annotated frame to a new output video. The result is a processed video where every detected object is labeled across all frames.

## Why Background Processing?
Video processing takes time — a 30-second video at 30fps has 900 frames, each requiring a YOLO inference pass. If the API waited for this to finish before responding, the client would time out. Background tasks solve this by:
- Returning a job ID immediately when the video is uploaded
- Processing the video in the background
- Letting the client check status at any time using the job ID
- Making the processed video available for download once complete

## Job/Status Workflow
```
Client uploads video
       ↓
API saves video, creates job ID, starts background processing
       ↓
Returns: { "job_id": "abc123", "status": "processing" }
       ↓
Client polls: GET /video/status/abc123
       ↓
Returns progress until status = "completed"
       ↓
Client downloads: GET /video/result/abc123
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Welcome message |
| GET | /health | Health check + model status |
| POST | /video/process | Upload video and start processing |
| GET | /video/status/{job_id} | Check processing status and progress |
| GET | /video/result/{job_id} | Download processed video |

## Error Handling

| Error | Status Code | Message |
|-------|-------------|---------|
| No file | 400 | No file provided |
| Wrong file type | 400 | Unsupported file type |
| Empty file | 400 | Uploaded file is empty |
| Job not found | 404 | Job not found |
| Job not complete | 400 | Job is not completed yet |
| Corrupted video | — | Job status set to failed with error message |

## Processing Performance
Results vary by video length and hardware. Typical CPU performance:

| Metric | Example |
|--------|---------|
| Total Frames | 300 |
| Processed Frames | 300 |
| Total Detections | 450 |
| Processing Time | ~45 sec |
| Average FPS | ~6-7 |

## Project Structure
```
Day-27/
├── app/
│   ├── main.py               — FastAPI app entry point
│   ├── routes/
│   │   └── video.py          — Upload, status, download endpoints
│   ├── services/
│   │   ├── detector.py       — YOLO model loading and frame inference
│   │   └── video_processor.py — Frame-by-frame processing and job tracking
│   └── utils/
│       └── file_utils.py     — File validation, job ID generation, path management
├── models/
│   └── best.pt               — YOLOv8n pretrained model
├── uploads/                  — Temporary uploaded videos
├── outputs/                  — Processed output videos
├── requirements.txt
└── README.md
```

## How to Run
```
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open Swagger UI: http://127.0.0.1:8000/docs

## Challenges Faced
- Background tasks in FastAPI run in the same process — for very long videos this can slow down other requests
- Video codec compatibility on Windows — used mp4v codec which works reliably with OpenCV
- Job state is stored in memory — restarting the server loses all job data

## Author
Hifsa Iftikhar