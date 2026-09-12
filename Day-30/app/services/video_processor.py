import cv2
import time
from pathlib import Path
from app.services.detector import detector
from app.utils.file_utils import get_output_path
from app.utils.logger import logger
from app.database.database import SessionLocal
from app.database import crud


def process_video(job_id: str, video_path: Path, confidence: float = 0.25):
    db = SessionLocal()
    try:
        logger.info(f"[{job_id}] Processing started - file: {video_path.name}")
        crud.update_job_processing(db, job_id)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"[{job_id}] Could not open video file.")
            crud.update_job_failed(db, job_id, "Could not open video file. File may be corrupted.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"[{job_id}] Video info - frames: {total_frames}, fps: {fps}, size: {width}x{height}")

        if total_frames == 0:
            logger.warning(f"[{job_id}] Video has no frames.")
            crud.update_job_failed(db, job_id, "Video file is empty or has no readable frames.")
            cap.release()
            return

        output_path = get_output_path(job_id)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        processed = 0
        total_detections = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, detections = detector.detect_frame(frame, confidence)
                total_detections += len(detections)
                out.write(annotated_frame)
                processed += 1

        except Exception as e:
            logger.error(f"[{job_id}] Processing failed at frame {processed}: {e}")
            crud.update_job_failed(db, job_id, f"Processing failed: {str(e)}")
            cap.release()
            out.release()
            return

        cap.release()
        out.release()

        elapsed = round(time.time() - start_time, 2)
        avg_fps = round(processed / elapsed, 2) if elapsed > 0 else 0

        stats = {
            "total_frames": total_frames,
            "processed_frames": processed,
            "total_detections": total_detections,
            "processing_time_sec": elapsed,
            "average_fps": avg_fps
        }

        crud.update_job_completed(db, job_id, stats, str(output_path))

        logger.info(
            f"[{job_id}] Completed - frames: {processed}, "
            f"detections: {total_detections}, time: {elapsed}s"
        )

    finally:
        db.close()