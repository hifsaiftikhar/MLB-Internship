from fastapi import (
    APIRouter,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Query,
    Depends
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional

from app.utils.file_utils import (
    validate_video_file,
    validate_confidence,
    generate_job_id,
    generate_request_id,
    get_upload_path,
    cleanup_upload
)
from app.services.video_processor import process_video
from app.database.database import get_db
from app.database import crud
from app.database.models import User
from app.services.auth import get_current_user
from app.utils.logger import logger

router = APIRouter(prefix="/video", tags=["Video Processing"])


def error_response(message: str, request_id: str, status_code: int):
    raise HTTPException(
        status_code=status_code,
        detail={
            "success": False,
            "error": message,
            "request_id": request_id
        }
    )


@router.post("/process")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    request_id = generate_request_id()

    logger.info(
        f"[{request_id}] User {current_user.username} "
        f"uploaded file: {file.filename}"
    )

    if not file.filename:
        error_response("No file provided.", request_id, 400)

    try:
        validate_video_file(file.filename)
    except ValueError as e:
        logger.warning(f"[{request_id}] {e}")
        error_response(str(e), request_id, 400)

    try:
        validate_confidence(confidence)
    except ValueError as e:
        logger.warning(f"[{request_id}] {e}")
        error_response(str(e), request_id, 400)

    contents = await file.read()

    if len(contents) == 0:
        logger.warning(f"[{request_id}] Empty file uploaded.")
        error_response("Uploaded file is empty.", request_id, 400)

    try:
        validate_video_file(file.filename, len(contents))
    except ValueError as e:
        logger.warning(f"[{request_id}] {e}")
        error_response(str(e), request_id, 400)

    job_id = generate_job_id()
    upload_path = get_upload_path(job_id, file.filename)

    with open(upload_path, "wb") as f:
        f.write(contents)

    # Important: save the job against the logged-in user
    crud.create_job(
        db,
        job_id,
        file.filename,
        current_user.id
    )

    background_tasks.add_task(
        process_video,
        job_id,
        upload_path,
        confidence
    )

    logger.info(
        f"[{request_id}] Job {job_id} created for user "
        f"{current_user.username}."
    )

    return {
        "success": True,
        "job_id": job_id,
        "request_id": request_id,
        "status": "queued",
        "message": f"Video '{file.filename}' uploaded. Processing started.",
        "check_status": f"/jobs/{job_id}"
    }


@router.get("/result/{job_id}")
def download_result(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    request_id = generate_request_id()

    job = crud.get_job(db, job_id)

    if not job:
        error_response(
            f"Job '{job_id}' not found.",
            request_id,
            404
        )

    # Normal users can only access their own jobs
    if current_user.role != "admin" and job.user_id != current_user.id:
        error_response(
            "You are not authorized to access this job.",
            request_id,
            403
        )

    if job.status != "completed":
        error_response(
            f"Job is not completed yet. Current status: {job.status}",
            request_id,
            400
        )

    output_path = Path(job.output_path)

    if not output_path.exists():
        error_response(
            "Processed video file not found on server.",
            request_id,
            404
        )

    logger.info(
        f"[{request_id}] Serving result for job: {job_id}"
    )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )


# =========================
# Jobs endpoints
# =========================

jobs_router = APIRouter(prefix="/jobs", tags=["Jobs"])


@jobs_router.get("/")
def get_all_jobs(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: queued, processing, completed, failed"
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Admin sees all jobs
    if current_user.role == "admin":
        jobs = crud.get_all_jobs(db, status=status)
    else:
        # Normal user sees only their own jobs
        jobs = crud.get_user_jobs(
            db,
            current_user.id,
            status=status
        )

    return {
        "success": True,
        "total": len(jobs),
        "role": current_user.role,
        "jobs": [
            {
                "job_id": j.job_id,
                "user_id": j.user_id,
                "filename": j.filename,
                "status": j.status,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
                "processing_time_sec": j.processing_time_sec,
                "total_detections": j.total_detections,
                "average_fps": j.average_fps
            }
            for j in jobs
        ]
    }


@jobs_router.get("/{job_id}")
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    request_id = generate_request_id()

    job = crud.get_job(db, job_id)

    if not job:
        error_response(
            f"Job '{job_id}' not found.",
            request_id,
            404
        )

    # Normal users can only view their own jobs
    if current_user.role != "admin" and job.user_id != current_user.id:
        error_response(
            "You are not authorized to access this job.",
            request_id,
            403
        )

    response = {
        "success": True,
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }

    if job.status == "completed":
        response["stats"] = {
            "processing_time_sec": job.processing_time_sec,
            "total_frames": job.total_frames,
            "processed_frames": job.processed_frames,
            "total_detections": job.total_detections,
            "average_fps": job.average_fps
        }

        response["download"] = f"/video/result/{job_id}"

    if job.status == "failed":
        response["success"] = False
        response["error"] = job.error

    return response


@jobs_router.delete("/{job_id}")
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    request_id = generate_request_id()

    job = crud.get_job(db, job_id)

    if not job:
        error_response(
            f"Job '{job_id}' not found.",
            request_id,
            404
        )

    # Normal users can delete only their own jobs
    if current_user.role != "admin" and job.user_id != current_user.id:
        error_response(
            "You are not authorized to delete this job.",
            request_id,
            403
        )

    cleanup_upload(job_id, job.filename)
    crud.delete_job(db, job_id)

    logger.info(
        f"User {current_user.username} deleted job {job_id}."
    )

    return {
        "success": True,
        "message": f"Job '{job_id}' deleted successfully."
    }