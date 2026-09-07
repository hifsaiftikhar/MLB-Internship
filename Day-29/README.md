# Day-29: AI Video Processing API with Database

## What Information is Stored in the Database
Every video processing job is stored permanently in SQLite with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| job_id | String | Unique 8-character job identifier |
| filename | String | Original uploaded video filename |
| status | String | queued, processing, completed, or failed |
| created_at | DateTime | When the job was created |
| completed_at | DateTime | When processing finished |
| processing_time_sec | Float | Total time taken to process |
| total_frames | Integer | Total frames in the video |
| processed_frames | Integer | Frames actually processed |
| total_detections | Integer | Total YOLO detections across all frames |
| average_fps | Float | Average frames processed per second |
| output_path | String | Path to the processed output video |
| error | String | Error message if job failed |

## Database Structure
Single table: `jobs`
- Primary key: `job_id`
- Database: SQLite (`jobs.db` file created automatically)
- ORM: SQLAlchemy

## What SQLAlchemy Does
SQLAlchemy is a Python library that lets you work with databases using Python classes instead of writing raw SQL. Instead of writing:
```sql
INSERT INTO jobs (job_id, filename, status) VALUES ('abc123', 'video.mp4', 'queued')
```
You write:
```python
job = Job(job_id="abc123", filename="video.mp4", status="queued")
db.add(job)
db.commit()
```
SQLAlchemy translates your Python code into SQL and handles the database connection.

## How the API Creates and Retrieves Jobs

**Creating a job:**
1. User uploads video to `POST /video/process`
2. API validates the file
3. Job record created in database with status "queued"
4. Background task starts processing
5. Database updated to "processing" → "completed" or "failed"

**Retrieving jobs:**
- `GET /jobs` returns all jobs from database
- `GET /jobs?status=completed` filters by status
- `GET /jobs/{job_id}` returns one specific job with full stats

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Welcome message |
| GET | /health | API and model status |
| POST | /video/process | Upload video and start processing |
| GET | /jobs | Get all jobs (filter by status) |
| GET | /jobs/{job_id} | Get specific job details |
| DELETE | /jobs/{job_id} | Delete a job record |
| GET | /video/result/{job_id} | Download processed video |

## Example API Requests and Responses

**POST /video/process:**
```json
Response 200:
{
  "success": true,
  "job_id": "a1b2c3d4",
  "status": "queued",
  "message": "Video 'traffic.mp4' uploaded. Processing started.",
  "check_status": "/jobs/a1b2c3d4"
}
```

**GET /jobs?status=completed:**
```json
{
  "success": true,
  "total": 2,
  "jobs": [
    {
      "job_id": "a1b2c3d4",
      "filename": "traffic.mp4",
      "status": "completed",
      "created_at": "2026-09-01T14:00:00",
      "completed_at": "2026-09-01T14:02:15",
      "processing_time_sec": 135.2,
      "total_detections": 450,
      "average_fps": 6.5
    }
  ]
}
```

**GET /jobs/{job_id} — completed:**
```json
{
  "success": true,
  "job_id": "a1b2c3d4",
  "filename": "traffic.mp4",
  "status": "completed",
  "stats": {
    "processing_time_sec": 135.2,
    "total_frames": 300,
    "processed_frames": 300,
    "total_detections": 450,
    "average_fps": 6.5
  },
  "download": "/video/result/a1b2c3d4"
}
```

**DELETE /jobs/{job_id}:**
```json
{
  "success": true,
  "message": "Job 'a1b2c3d4' deleted successfully."
}
```

## Difference from Day-28
| | Day-28 | Day-29 |
|--|--------|--------|
| Storage | In-memory dictionary | SQLite database |
| Survives restart | No | Yes |
| Query by status | No | Yes |
| Delete jobs | No | Yes |
| Job history | Lost on restart | Permanent |

## Project Structure
```
Day-29/
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── video.py
│   ├── services/
│   │   ├── detector.py
│   │   └── video_processor.py
│   ├── utils/
│   │   ├── file_utils.py
│   │   └── logger.py
│   └── database/
│       ├── database.py
│       ├── models.py
│       └── crud.py
├── models/
│   └── best.pt
├── uploads/
├── outputs/
├── logs/
├── jobs.db
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