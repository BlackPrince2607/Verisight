from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.db.session import get_db
from app.api.models.models import AnalysisJob

router = APIRouter()


@router.get("/{job_id}")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter_by(job_id=job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response = {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "result": job.result,
        "confidence": job.confidence,
        "created_at": job.created_at,
    }

    # Add human-readable verdict only when completed
    if job.status == "completed" and job.result:
        response["verdict"] = "FAKE" if "fake" in job.result.lower() else "REAL"

    return response