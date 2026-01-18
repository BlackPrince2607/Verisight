from fastapi import FastAPI
from apps.api.routes import upload, jobs, results

app = FastAPI(title="VeriSight API")

@app.get("/")
def root():
    return {"message": "VeriSight API running"}

# Register routes
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
app.include_router(results.router, prefix="/results", tags=["Results"])
