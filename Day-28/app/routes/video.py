from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from app.utils.file_utils import (
    validate_video_file, validate_confidence,
    generate_job_id, generate_request_id,
    get_upload_path
)
from app.services.video_processor import process_video, create_job, get_job
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
    confidence: float = Query(default=0.25, ge=0.01, le=1.0)
):
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Video upload received - file: {file.filename}")

    # Validate filename
    if not file.filename:
        logger.warning(f"[{request_id}] Upload rejected - no filename")
        error_response("No file provided.", request_id, 400)

    # Validate file type
    try:
        validate_video_file(file.filename)
    except ValueError as e:
        logger.warning(f"[{request_id}] Upload rejected - {e}")
        error_response(str(e), request_id, 400)

    # Validate confidence
    try:
        validate_confidence(confidence)
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid confidence - {e}")
        error_response(str(e), request_id, 400)

    # Read and validate file size
    contents = await file.read()
    if len(contents) == 0:
        logger.warning(f"[{request_id}] Upload rejected - empty file")
        error_response("Uploaded file is empty.", request_id, 400)

    try:
        validate_video_file(file.filename, len(contents))
    except ValueError as e:
        logger.warning(f"[{request_id}] Upload rejected - {e}")
        error_response(str(e), request_id, 400)

    # Save file and create job
    job_id = generate_job_id()
    upload_path = get_upload_path(job_id, file.filename)

    with open(upload_path, "wb") as f:
        f.write(contents)

    create_job(job_id, file.filename)
    background_tasks.add_task(process_video, job_id, upload_path, confidence)

    logger.info(f"[{request_id}] Job {job_id} created and queued.")

    return {
        "success": True,
        "job_id": job_id,
        "request_id": request_id,
        "status": "processing",
        "message": f"Video '{file.filename}' uploaded successfully. Processing started.",
        "check_status": f"/video/status/{job_id}"
    }


@router.get("/status/{job_id}")
def get_status(job_id: str):
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Status check for job: {job_id}")

    job = get_job(job_id)
    if not job:
        logger.warning(f"[{request_id}] Job not found: {job_id}")
        error_response(f"Job '{job_id}' not found.", request_id, 404)

    response = {
        "success": True,
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job["filename"],
    }

    if job["status"] == "completed":
        response["stats"] = job["stats"]
        response["download"] = f"/video/result/{job_id}"

    if job["status"] == "failed":
        response["success"] = False
        response["error"] = job["error"]

    return response


@router.get("/result/{job_id}")
def download_result(job_id: str):
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Download requested for job: {job_id}")

    job = get_job(job_id)
    if not job:
        logger.warning(f"[{request_id}] Job not found: {job_id}")
        error_response(f"Job '{job_id}' not found.", request_id, 404)

    if job["status"] != "completed":
        logger.warning(f"[{request_id}] Download attempted on incomplete job: {job_id}")
        error_response(
            f"Job is not completed yet. Current status: {job['status']}",
            request_id, 400
        )

    output_path = Path(job["output_path"])
    if not output_path.exists():
        logger.error(f"[{request_id}] Output file missing for job: {job_id}")
        error_response("Processed video file not found on server.", request_id, 404)

    logger.info(f"[{request_id}] Serving result for job: {job_id}")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )