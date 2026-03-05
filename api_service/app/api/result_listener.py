import json
import threading
import redis
from sqlalchemy.orm import Session
from app.api.db.session import SessionLocal
from app.api.models.models import AnalysisJob

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def listen_for_results():
    print("Result listener started...")
    while True:
        try:
            result_data = redis_client.brpop("analysis_results", timeout=5)
            if not result_data:
                continue

            result_message = json.loads(result_data[1])

            job_id = result_message["job_id"]
            status = result_message["status"]
            result = result_message.get("result")
            confidence = result_message.get("confidence")

            db: Session = SessionLocal()
            try:
                job = db.query(AnalysisJob).filter_by(job_id=job_id).first()
                if job:
                    job.status = status
                    job.result = result
                    job.confidence = confidence
                    db.commit()
                else:
                    print(f"[WARN] Job {job_id} not found in DB")
            except Exception as db_err:
                db.rollback()
                print(f"[ERROR] DB write failed for job {job_id}: {db_err}")
            finally:
                db.close()

        except json.JSONDecodeError as e:
            print(f"[ERROR] Bad JSON from Redis: {e}")
        except Exception as e:
            print(f"[ERROR] Listener crashed, restarting loop: {e}")


def start_listener():
    thread = threading.Thread(target=listen_for_results, daemon=True)
    thread.start()