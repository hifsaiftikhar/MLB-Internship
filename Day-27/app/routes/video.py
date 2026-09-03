from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from app.utils.file_utils import validate_video_file, generate_job_id, get_upload_path
from app.services.video_processor import process_video, create_job, get_job

router = APIRouter(prefix="/video", tags=["Video Processing"])


@router.post("/process")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0)
):
    """
    Upload a video and start YOLO processing in the background.
    Returns a job_id to track progress.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    try:
        validate_video_file(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Read file contents
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Generate job ID and save upload
    job_id = generate_job_id()
    upload_path = get_upload_path(job_id, file.filename)

    with open(upload_path, "wb") as f:
        f.write(contents)

    # Initialize job and start background processing
    create_job(job_id, file.filename)
    background_tasks.add_task(process_video, job_id, upload_path, confidence)

    return {
        "job_id": job_id,
        "status": "processing",
        "message": f"Video '{file.filename}' uploaded. Processing started.",
        "check_status": f"/video/status/{job_id}"
    }


@router.get("/status/{job_id}")
def get_status(job_id: str):
    """
    Check the processing status of a job.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job["filename"],
    }

    if job["status"] == "completed":
        response["stats"] = job["stats"]
        response["download"] = f"/video/result/{job_id}"

    if job["status"] == "failed":
        response["error"] = job["error"]

    return response


@router.get("/result/{job_id}")
def download_result(job_id: str):
    """
    Download the processed video once job is completed.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet. Current status: {job['status']}"
        )

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video file not found.")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )