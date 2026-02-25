import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.db.session import get_db
from app.api.models.models import AnalysisJob
from app.api.core.redis_client import redis_client
import json

router = APIRouter()

UPLOAD_DIR = "uploads/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Save file
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Create DB record
        new_job = AnalysisJob(
            job_id=job_id,
            filename=file.filename,
            file_path=file_path,
            status="queued"
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        job_message = {
            "job_id": job_id,
            "file_path": file_path,
            "media_type": "image",
            "model_version": "v1"
        }
        print("Pushing to Redis:", job_message)
        redis_client.lpush("analysis_jobs", json.dumps(job_message))

        return {
            "message": "File uploaded successfully",
            "job_id": job_id,
            "status": "queued"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))