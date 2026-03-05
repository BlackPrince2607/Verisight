from fastapi import FastAPI
from app.api.routes import upload, jobs, results
from app.api.db.session import engine, Base
from app.api.models import models
from app.api.result_listener import start_listener
from app.api.routes import upload, status
from app.api.core.redis_client import redis_client
from app.api.db.session import SessionLocal
from sqlalchemy import text
from app.api.routes import upload, status, stream
from fastapi import StaticFiles

import os
app = FastAPI(title="VeriSight API")
Base.metadata.create_all(bind=engine)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_UPLOAD_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "uploads", "images"))
app.mount("/static", StaticFiles(directory="app/static"), name="static")
@app.get("/")
def root():
    return {"message": "VeriSight API running"}
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "uptime": "operational"}
@app.get("/health/details", tags=["Health"])
def health_details():
    redis_ok = False
    db_ok = False

    try:
        redis_client.ping()
        redis_ok = True
    except Exception as e:
        print(f"[HEALTH] Redis check failed: {e}")

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
        db.close()
    except Exception as e:
        print(f"[HEALTH] DB check failed: {e}")

    return {
        "redis":    redis_ok,
        "database": db_ok,
        "worker":   redis_ok  # if Redis is reachable, worker can connect
    }

# Register routes
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
app.include_router(results.router, prefix="/results", tags=["Results"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(status.router, prefix="/status", tags=["Status"])
app.include_router(stream.router, tags=["Stream"])
@app.on_event("startup")
def startup_event():
    start_listener()
    