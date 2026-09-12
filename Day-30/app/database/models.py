from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # "user" or "admin"
    created_at = Column(DateTime, default=func.now())

    jobs = relationship("Job", back_populates="owner")


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
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

    owner = relationship("User", back_populates="jobs")