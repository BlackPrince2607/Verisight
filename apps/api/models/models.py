from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True)
    job_id = Column(String(36), unique=True)
    filename = Column(String(255))
    status = Column(String(50), default="queued")
    created_at = Column(DateTime, default=datetime.utcnow)
