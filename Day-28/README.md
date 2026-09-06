# Day-28: Production-Ready AI Video Processing API

## What Validations Were Added

| Validation | Rule | Error Returned |
|------------|------|----------------|
| File type | Only .mp4, .avi, .mov, .mkv, .webm | 400 Unsupported file format |
| File size | Maximum 200 MB | 400 File too large |
| Empty file | File must have content | 400 Uploaded file is empty |
| Confidence | Must be between 0.01 and 1.0 | 400 Invalid confidence value |
| Job ID | Must exist in job store | 404 Job not found |
| Job status | Must be completed before download | 400 Job not completed yet |

## Error Handling
Every error returns a structured JSON response:
```json
{
  "success": false,
  "error": "Unsupported file format '.txt'. Allowed: .avi, .mkv, .mov, .mp4, .webm",
  "request_id": "req_a1b2c3d4"
}
```

A global exception handler catches any unexpected server error and returns a 500 response instead of crashing the API.

## What Gets Logged
Every important event is recorded in `logs/app.log`:

```
2026-09-01 14:00:01 | INFO     | Video upload received - file: traffic.mp4
2026-09-01 14:00:01 | INFO     | Job abc123 created and queued
2026-09-01 14:00:01 | INFO     | Processing started - file: traffic.mp4
2026-09-01 14:00:01 | INFO     | Video info - frames: 300, fps: 30, size: 1280x720
2026-09-01 14:02:15 | INFO     | Processing completed - frames: 300, detections: 450, time: 134s
2026-09-01 14:05:00 | WARNING  | Upload rejected - Unsupported file format '.txt'
2026-09-01 14:06:00 | WARNING  | Job not found: fake123
2026-09-01 14:07:00 | ERROR    | Could not open video file - file may be corrupted
```

Sensitive information (file contents, personal data) is never logged.

## HTTP Status Codes Used

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | OK | Successful request |
| 400 | Bad Request | Validation failed |
| 404 | Not Found | Job ID not found, file missing |
| 500 | Server Error | Unexpected error caught by global handler |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Welcome message |
| GET | /health | API status, model status, version |
| POST | /video/process | Upload video, start processing |
| GET | /video/status/{job_id} | Check job progress |
| GET | /video/result/{job_id} | Download processed video |

## 3 Examples of Failed Requests

**1. Wrong file type:**
```json
POST /video/process (uploaded: document.txt)
Response 400:
{
  "success": false,
  "error": "Unsupported file format '.txt'. Allowed: .avi, .mkv, .mov, .mp4, .webm",
  "request_id": "req_a1b2c3d4"
}
```

**2. Empty file:**
```json
POST /video/process (uploaded: empty.mp4 - 0 bytes)
Response 400:
{
  "success": false,
  "error": "Uploaded file is empty.",
  "request_id": "req_b2c3d4e5"
}
```

**3. Invalid job ID:**
```json
GET /video/status/fakeid99
Response 404:
{
  "success": false,
  "error": "Job 'fakeid99' not found.",
  "request_id": "req_c3d4e5f6"
}
```

## Project Structure
```
Day-28/
├── app/
│   ├── main.py                  — FastAPI app with global exception handler
│   ├── routes/
│   │   └── video.py             — Upload, status, download endpoints
│   ├── services/
│   │   ├── detector.py          — YOLO model with logging
│   │   └── video_processor.py   — Frame processing with job tracking and logging
│   └── utils/
│       ├── file_utils.py        — Validation and path management
│       └── logger.py            — Structured logger setup
├── models/
│   └── best.pt
├── uploads/
├── outputs/
├── logs/
│   └── app.log
├── requirements.txt
└── README.md
```

## How to Run
```
pip install -r requirements.txt
uvicorn app.main:app
```

## Author
Hifsa Iftikhar