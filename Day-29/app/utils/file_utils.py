import uuid
import os
from pathlib import Path

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 200

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def generate_job_id() -> str:
    return str(uuid.uuid4())[:8]


def generate_request_id() -> str:
    return "req_" + str(uuid.uuid4())[:8]


def validate_video_file(filename: str, file_size: int = None) -> None:
    if not filename or filename.strip() == "":
        raise ValueError("No filename provided. Please upload a valid video file.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    if file_size is not None:
        size_mb = file_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            )


def validate_confidence(confidence: float) -> None:
    if not (0.01 <= confidence <= 1.0):
        raise ValueError(
            f"Confidence threshold must be between 0.01 and 1.0. Got: {confidence}"
        )


def get_upload_path(job_id: str, filename: str) -> Path:
    ext = Path(filename).suffix.lower()
    return UPLOADS_DIR / f"{job_id}{ext}"


def get_output_path(job_id: str) -> Path:
    return OUTPUTS_DIR / f"{job_id}_processed.mp4"


def cleanup_upload(job_id: str, filename: str) -> None:
    upload_path = get_upload_path(job_id, filename)
    if upload_path.exists():
        os.remove(upload_path)