# VeriSight Copilot Instructions

## Project Architecture

**VeriSight** is an async image analysis platform using FastAPI, MySQL, and Redis for job queue management.

**Core Flow:**
1. Client uploads image → FastAPI endpoint generates UUID-based job → stores in MySQL → pushes job ID to Redis queue
2. Background worker polls Redis queue → processes job → updates DB status (queued → processing → completed/failed)
3. Client queries `/api/jobs` endpoint to fetch results

**Key Components:**
- [app/main.py](app/main.py) - FastAPI application entry point, route registration
- [app/db/session.py](app/db/session.py) - SQLAlchemy engine and session factory using mysql+pymysql
- [app/db/base.py](app/db/base.py) - Database initialization (replaces scripts/create_tables.py)
- [app/models/analysis_job.py](app/models/analysis_job.py) - `AnalysisJob` ORM model with UUID tracking and status states
- [app/api/routes/jobs.py](app/api/routes/jobs.py) - Image upload and job lifecycle management
- [app/core/redis.py](app/core/redis.py) - Redis client initialization and queue operations
- [worker/worker.py](worker/worker.py) - Long-running daemon pulling jobs from Redis, updating DB

## Environment & Dependencies

**Config via `.env`** (loaded by `python-dotenv`):
- `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` - MySQL credentials
- `REDIS_URL` - defaults to `redis://localhost:6379/0`

**Core Stack:**
- FastAPI (async endpoints)
- SQLAlchemy ORM (MySQL 8 via PyMySQL)
- Redis (job queue with `brpop` blocking operations)
- Python-dotenv (environment config)

## Developer Workflows

### Setup & Run

```bash
# Initialize database tables
python -c "from app.db.base import Base, engine; Base.metadata.create_all(bind=engine)"

# Start API server
python -m uvicorn app.main:app --reload --port 8000

# Start worker (separate terminal)
python worker/worker.py
```

### Key Patterns

**Database Session Management:**
- Uses `SessionLocal = sessionmaker(bind=engine)` from [app/db/session.py](app/db/session.py)
- Session dependency injection: `Depends(get_db)` in route handlers
- Always close sessions in worker finally blocks to prevent leaks

**Job Lifecycle States:**
- `queued` → `processing` → `completed` or `failed`
- Worker catches exceptions and commits failure state before closing session

**Redis Queue Pattern:**
- `lpush()` adds jobs (server-side), `brpop()` retrieves with timeout (worker-side)
- Queue name: `"verisight_image_jobs"`
- Job data stored as job ID integers from database

**File Storage:**
- Images saved to `uploads/images/` with UUID-based filenames
- Filename format: `{uuid}_{original_filename}`

## When Modifying This Codebase

1. **Adding new routes:** Create new route module in [app/api/routes/](app/api/routes/), register in [app/main.py](app/main.py)
2. **Database changes:** Update [app/models/analysis_job.py](app/models/analysis_job.py), re-run init command
3. **Async job processing:** All Redis operations are non-blocking via `brpop()` with timeout; safe for long operations
4. **Worker enhancements:** Implement actual analysis logic in [worker/worker.py](worker/worker.py) (currently uses dummy 5s delay for testing)

## Testing & Quality

- Tests are in [tests/](tests/) but minimal coverage currently
- No linting/type-checking configured; consider adding mypy, black, flake8
- Worker error handling is basic; production should implement retry logic and DLQ (dead-letter queue)

## Critical Implementation Notes

- **Security:** [app/core/security.py](app/core/security.py) exists but is empty; add CORS, auth as needed
- **File validation:** Only MIME-type check for images; consider adding file size limits, virus scanning
- **Worker processing:** Replace `time.sleep(5)` mock with actual analysis logic
- **Error recovery:** No automatic retry on worker failure; implement exponential backoff if needed
