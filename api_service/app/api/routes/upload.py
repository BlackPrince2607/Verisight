import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.db.session import get_db
from app.api.models.models import AnalysisJob
from app.api.routes import upload, status
from app.api.core.redis_client import redis_client
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "..", "..", "uploads", "images")
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)  # resolves to clean absolute path
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    try:
        job_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

        # Async file write — doesn't block event loop
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(contents)

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
        redis_client.lpush("analysis_jobs", json.dumps(job_message))

        return {
            "message": "File uploaded successfully",
            "job_id": job_id,
            "status": "queued"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 