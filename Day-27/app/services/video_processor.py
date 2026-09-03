import cv2
import time
from pathlib import Path
from app.services.detector import detector
from app.utils.file_utils import get_output_path

# In-memory job store
jobs: dict = {}


def process_video(job_id: str, video_path: Path, confidence: float = 0.25):
    jobs[job_id]["status"] = "processing"
    output_path = get_output_path(job_id)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Could not open video file."
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Video has no frames."
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    processed = 0
    total_detections = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, detections = detector.detect_frame(frame, confidence)
        total_detections += len(detections)
        out.write(annotated_frame)

        processed += 1
        progress = int((processed / total_frames) * 100)
        jobs[job_id]["progress"] = progress
        jobs[job_id]["processed_frames"] = processed

    cap.release()
    out.release()

    elapsed = round(time.time() - start_time, 2)
    avg_fps = round(processed / elapsed, 2) if elapsed > 0 else 0

    jobs[job_id].update({
        "status": "completed",
        "progress": 100,
        "output_path": str(output_path),
        "stats": {
            "total_frames": total_frames,
            "processed_frames": processed,
            "total_detections": total_detections,
            "processing_time_sec": elapsed,
            "average_fps": avg_fps
        }
    })


def create_job(job_id: str, filename: str):
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "filename": filename,
        "processed_frames": 0,
        "output_path": None,
        "error": None,
        "stats": None
    }


def get_job(job_id: str):
    return jobs.get(job_id)