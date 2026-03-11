import os
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.db.session import engine, Base, SessionLocal, get_db
from app.api.models import models
from app.api.result_listener import start_listener
from app.api.core.redis_client import redis_client
from app.api.routes import upload, status, stream, jobs, results

app = FastAPI(title="VeriSight API")

# DB init
Base.metadata.create_all(bind=engine)

# Static files — absolute path
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Root → dashboard
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")

# Routers — each registered once
app.include_router(upload.router,  prefix="/upload",  tags=["Upload"])
app.include_router(status.router,  prefix="/status",  tags=["Status"])
app.include_router(stream.router,  tags=["Stream"])
app.include_router(jobs.router,    prefix="/jobs",    tags=["Jobs"])
app.include_router(results.router, prefix="/results", tags=["Results"])

# Health
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.get("/health/details", tags=["Health"])
def health_details():
    redis_ok = db_ok = False
    try:
        redis_client.ping()
        redis_ok = True
    except: pass
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
        db.close()
    except: pass
    return {"redis": redis_ok, "database": db_ok, "worker": redis_ok}
    
@app.get("/stats", tags=["Stats"])
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    from app.api.models.models import AnalysisJob

    total = db.query(func.count(AnalysisJob.id)).scalar() or 0

    completed = db.query(AnalysisJob).filter(
        AnalysisJob.status == "completed"
    ).all()

    if not completed:
        return {
            "total_analyzed": total,
            "avg_confidence": 0,
            "fake_percentage": 0,
        }

    confidences = [j.confidence for j in completed if j.confidence]
    avg_conf    = round(sum(confidences) / len(confidences) * 100) if confidences else 0
    fake_count  = sum(1 for j in completed if "fake" in (j.result or "").lower())
    fake_pct    = round(fake_count / len(completed) * 100) if completed else 0

    return {
        "total_analyzed": total,
        "avg_confidence":  avg_conf,
        "fake_percentage": fake_pct,
    }
# Startup
@app.on_event("startup")
def startup_event():
    start_listener()