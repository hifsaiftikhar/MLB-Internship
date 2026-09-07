from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from typing import Optional, List
from app.database.models import Job


def create_job(db: Session, job_id: str, filename: str) -> Job:
    """Create a new job record in the database."""
    job = Job(
        job_id=job_id,
        filename=filename,
        status="queued",
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    """Get a job by ID."""
    return db.query(Job).filter(Job.job_id == job_id).first()


def get_all_jobs(db: Session, status: Optional[str] = None) -> List[Job]:
    """Get all jobs, optionally filtered by status."""
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(Job.created_at.desc()).all()


def update_job_processing(db: Session, job_id: str) -> Optional[Job]:
    """Mark job as processing."""
    job = get_job(db, job_id)
    if job:
        job.status = "processing"
        db.commit()
        db.refresh(job)
    return job


def update_job_completed(db: Session, job_id: str, stats: dict, output_path: str) -> Optional[Job]:
    """Mark job as completed and store results."""
    job = get_job(db, job_id)
    if job:
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.processing_time_sec = stats.get("processing_time_sec")
        job.total_frames = stats.get("total_frames")
        job.processed_frames = stats.get("processed_frames")
        job.total_detections = stats.get("total_detections")
        job.average_fps = stats.get("average_fps")
        job.output_path = output_path
        db.commit()
        db.refresh(job)
    return job


def update_job_failed(db: Session, job_id: str, error: str) -> Optional[Job]:
    """Mark job as failed with error message."""
    job = get_job(db, job_id)
    if job:
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        job.error = error
        db.commit()
        db.refresh(job)
    return job


def delete_job(db: Session, job_id: str) -> bool:
    """Delete a job record."""
    job = get_job(db, job_id)
    if job:
        db.delete(job)
        db.commit()
        return True
    return False