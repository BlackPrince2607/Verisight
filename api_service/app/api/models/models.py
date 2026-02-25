from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from datetime import datetime
from app.api.db.session import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True)
    
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)

    status = Column(String(50), default="queued", index=True)

    result = Column(String(50), nullable=True)         # real / fake
    confidence = Column(Float, nullable=True)         # probability score
    error_message = Column(Text, nullable=True)       # store failure reason

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)