import json
import threading
import redis
from sqlalchemy.orm import Session
from app.api.db.session import SessionLocal
from app.api.models.models import AnalysisJob

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def listen_for_results():
    while True:
        result_data = redis_client.brpop("analysis_results")
        result_message = json.loads(result_data[1])

        job_id = result_message["job_id"]
        status = result_message["status"]
        result = result_message["result"]
        confidence = result_message["confidence"]

        db: Session = SessionLocal()

        job = db.query(AnalysisJob).filter_by(job_id=job_id).first()
        if job:
            job.status = status
            job.result = result
            job.confidence = confidence
            db.commit()

        db.close()


def start_listener():
    thread = threading.Thread(target=listen_for_results, daemon=True)
    thread.start()