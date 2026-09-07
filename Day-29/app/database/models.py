from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.sql import func
from app.database.database import Base


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="queued")
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    processing_time_sec = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, nullable=True)
    total_detections = Column(Integer, nullable=True)
    average_fps = Column(Float, nullable=True)
    output_path = Column(String, nullable=True)
    error = Column(String, nullable=True)