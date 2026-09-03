import uuid
import os
from pathlib import Path

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())[:8]


def validate_video_file(filename: str, file_size: int = None) -> None:
    """
    Validate video file extension and size.
    Raises ValueError with a clear message if invalid.
    """
    if not filename:
        raise ValueError("No filename provided.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    if file_size is not None:
        size_mb = file_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            )


def get_upload_path(job_id: str, filename: str) -> Path:
    """Get the path to save an uploaded video."""
    ext = Path(filename).suffix.lower()
    return UPLOADS_DIR / f"{job_id}{ext}"


def get_output_path(job_id: str) -> Path:
    """Get the path for the processed output video."""
    return OUTPUTS_DIR / f"{job_id}_processed.mp4"


def cleanup_upload(job_id: str, filename: str) -> None:
    """Delete the uploaded file after processing."""
    upload_path = get_upload_path(job_id, filename)
    if upload_path.exists():
        os.remove(upload_path)