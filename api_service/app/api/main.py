from fastapi import FastAPI
from app.api.routes import upload, jobs, results
from app.api.db.session import engine, Base
from app.api.models import models
from app.api.result_listener import start_listener
app = FastAPI(title="VeriSight API")
Base.metadata.create_all(bind=engine)

@app.on_event("startup")
def startup_event():
    start_listener()
    
@app.get("/")
def root():
    return {"message": "VeriSight API running"}

# Register routes
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
app.include_router(results.router, prefix="/results", tags=["Results"])
