import os
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles          # ✅ correct import
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.db.session import engine, Base, SessionLocal, get_db
from app.api.models import models
from app.api.models.models import AnalysisJob
from app.api.result_listener import start_listener
from app.api.core.redis_client import redis_client
from app.api.routes import upload, status, stream, jobs, results

app = FastAPI(title="VeriSight API")

# ── DB init ──────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Static files (absolute path) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Root redirect ─────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")

# ── Routers (each registered ONCE) ───────────────────────
app.include_router(upload.router,  prefix="/upload",  tags=["Upload"])
app.include_router(status.router,  prefix="/status",  tags=["Status"])
app.include_router(stream.router,  tags=["Stream"])
app.include_router(jobs.router,    prefix="/jobs",    tags=["Jobs"])
app.include_router(results.router, prefix="/results", tags=["Results"])

# ── Health ────────────────────────────────────────────────
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

    return {"redis": redis_ok, "database": db_ok, "worker": redis_ok}

# ── Jobs history ──────────────────────────────────────────
@app.get("/jobs-list", tags=["Jobs"])
def get_jobs(limit: int = 20, db: Session = Depends(get_db)):
    jobs_list = db.query(AnalysisJob)\
                  .order_by(AnalysisJob.created_at.desc())\
                  .limit(limit).all()
    return [
        {
            "job_id":     j.job_id,
            "filename":   j.filename,
            "status":     j.status,
            "result":     j.result,
            "verdict":    j.result,
            "confidence": j.confidence,
            "media_type": getattr(j, "media_type", "image"),
            "created_at": str(j.created_at),
        }
        for j in jobs_list
    ]

# ── Startup ───────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    start_listener()