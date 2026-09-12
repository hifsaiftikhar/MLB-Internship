from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Optional, List
import os
from app.database.models import User, Job
from app.services.auth import hash_password


# ===== User CRUD =====

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, email: str, password: str, role: str = "user") -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        created_at=datetime.now(timezone.utc)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ===== Job CRUD =====

def create_job(db: Session, job_id: str, filename: str, user_id: int) -> Job:
    job = Job(
        job_id=job_id,
        filename=filename,
        status="queued",
        user_id=user_id,
        created_at=datetime.now(timezone.utc)
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    return db.query(Job).filter(Job.job_id == job_id).first()


def get_user_jobs(db: Session, user_id: int, status: Optional[str] = None) -> List[Job]:
    query = db.query(Job).filter(Job.user_id == user_id)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(Job.created_at.desc()).all()


def get_all_jobs(db: Session, status: Optional[str] = None) -> List[Job]:
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(Job.created_at.desc()).all()


def update_job_processing(db: Session, job_id: str) -> Optional[Job]:
    job = get_job(db, job_id)
    if job:
        job.status = "processing"
        db.commit()
        db.refresh(job)
    return job


def update_job_completed(db: Session, job_id: str, stats: dict, output_path: str) -> Optional[Job]:
    job = get_job(db, job_id)
    if job:
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
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
    job = get_job(db, job_id)
    if job:
        job.status = "failed"
        job.completed_at = datetime.now(timezone.utc)
        job.error = error
        db.commit()
        db.refresh(job)
    return job


def delete_job(db: Session, job_id: str) -> bool:
    job = get_job(db, job_id)
    if job:
        if job.output_path and os.path.exists(job.output_path):
            try:
                os.remove(job.output_path)
            except Exception:
                pass
        db.delete(job)
        db.commit()
        return True
    return False