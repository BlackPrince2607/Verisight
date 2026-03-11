from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.db.session import get_db
from app.api.models.models import AnalysisJob

router = APIRouter()


@router.get("/")
def get_jobs(limit: int = 20, db: Session = Depends(get_db)):
    jobs = db.query(AnalysisJob)\
             .order_by(AnalysisJob.created_at.desc())\
             .limit(limit)\
             .all()
    return [
        {
            "job_id":     j.job_id,
            "filename":   j.filename,
            "status":     j.status,
            "result":     j.result,
            "confidence": j.confidence,
            "created_at": str(j.created_at),
        }
        for j in jobs
    ]