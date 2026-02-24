import time
import redis
from sqlalchemy.orm import Session
from app.api.db.session import SessionLocal
from app.api.models.models import AnalysisJob

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

QUEUE_NAME = "analysis_queue"


def process_job(job_id: str):
    db: Session = SessionLocal()

    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.job_id == job_id).first()

        if not job:
            print(f"Job {job_id} not found in DB")
            return

        print(f"Processing job {job_id}")

        # Update status to processing
        job.status = "processing"
        db.commit()

        # Simulate AI processing
        time.sleep(5)

        # Fake result (we will replace with real AI later)
        job.status = "completed"
        job.result = "real"
        job.confidence = 0.95

        db.commit()

        print(f"Completed job {job_id}")

    except Exception as e:
        print(f"Error processing job {job_id}: {e}")

    finally:
        db.close()


def start_worker():
    print("Worker started. Waiting for jobs...")

    while True:
        job_data = redis_client.brpop(QUEUE_NAME)

        if job_data:
            _, job_id = job_data
            process_job(job_id)


if __name__ == "__main__":
    start_worker()