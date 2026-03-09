# VeriSight — Deepfake Detection System

> AI-powered media authentication system for detecting deepfake images, videos, and live streams using deep learning and distributed microservice architecture.

---

## Overview

VeriSight is a production-style backend system that classifies media as **Real** or **Fake** using a fine-tuned CNN model. It processes uploaded images and videos asynchronously through a Redis-based job pipeline, and supports real-time deepfake detection on live streams (YouTube, RTMP, RTSP, IP cameras) via WebSocket.

The system is split into two independent microservices — an API service and an inference service — communicating over HTTP and Redis, designed to scale independently.

---

## Features

- **Deepfake Detection** — CNN-based classification of images and video frames as Real or Fake with confidence scores
- **Face Detection Pipeline** — OpenCV face extraction before inference; model analyzes the face region rather than the full frame, improving accuracy
- **Asynchronous Job Processing** — Upload jobs are queued in Redis and processed by the inference worker without blocking the API
- **Live Stream Detection** — Real-time deepfake analysis on YouTube live streams, RTMP, RTSP, and HLS sources via WebSocket
- **Confidence Smoothing** — Rolling weighted average over 5 frames prevents flickering verdicts on live streams
- **Result Persistence** — All analysis jobs tracked in MySQL with status, verdict, confidence, and timestamps
- **System Health Monitoring** — API endpoints expose Redis, database, and worker status
- **Web Dashboard** — Static frontend for uploading media, viewing live results, and browsing job history

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Browser                          │
│          Dashboard (index.html / stream.html)           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / WebSocket
                         ▼
┌─────────────────────────────────────────────────────────┐
│              API Service  :8000  (FastAPI)               │
│                                                         │
│  POST /upload/          →  save file + queue job        │
│  GET  /status/{job_id}  →  poll result from DB          │
│  GET  /jobs-list        →  job history                  │
│  GET  /health/details   →  service status               │
│  WS   /ws/stream        →  live stream WebSocket        │
│         │                        │                      │
│         │ lpush                  │ POST /stream/start   │
│         ▼                        ▼                      │
│      Redis                 Inference Service            │
│   analysis_jobs              :8001                      │
│   analysis_results                                      │
│   stream_results                                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Inference Service  :8001  (FastAPI)            │
│                                                         │
│  Background Thread: polls analysis_jobs from Redis      │
│    → OpenCV face detection                              │
│    → CNN inference (GPU if available)                   │
│    → pushes result to analysis_results                  │
│                                                         │
│  POST /stream/start  →  starts stream_analysis_loop     │
│  POST /stream/stop   →  cancels stream task             │
│  GET  /health        →  device + model info             │
│                                                         │
│  Stream loop: OpenCV VideoCapture → frame every 10th    │
│    → face detection → inference → push to stream_results│
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                     MySQL Database                       │
│              Table: analysis_jobs                        │
│   job_id · filename · status · result · confidence      │
│   fake_frame_ratio · total_frames_analyzed · created_at │
└─────────────────────────────────────────────────────────┘
```

### Design Patterns

- **Message Broker Pattern** — Redis decouples the API and inference services; neither depends on the other being available at the same time
- **Microservice Architecture** — API and inference run as separate processes on separate ports, independently deployable
- **Asynchronous Job Processing** — uploads return immediately with a job ID; clients poll `/status/{job_id}` for results
- **Confidence-Weighted Aggregation** — video and stream verdicts weight each frame's vote by its confidence score rather than a simple majority

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Inference Framework | PyTorch + HuggingFace Transformers |
| Model | `prithivMLmods/Deep-Fake-Detector-Model` (FaceForensics++) |
| Face Detection | OpenCV Haar Cascade |
| Job Queue | Redis |
| Database | MySQL + SQLAlchemy ORM |
| Stream Ingestion | OpenCV VideoCapture + yt-dlp |
| Async Transport | WebSocket (FastAPI) + httpx |
| GPU Inference | CUDA (falls back to CPU) |

---

## Project Structure

```
verisight/
├── api_service/
│   └── app/
│       ├── api/
│       │   ├── main.py               # FastAPI app, routers, health endpoints
│       │   ├── core/
│       │   │   └── redis_client.py   # shared Redis connection
│       │   ├── db/
│       │   │   └── session.py        # SQLAlchemy session
│       │   ├── models/
│       │   │   └── models.py         # AnalysisJob ORM model
│       │   ├── routes/
│       │   │   ├── upload.py         # POST /upload/
│       │   │   ├── status.py         # GET /status/{job_id}
│       │   │   ├── stream.py         # WS /ws/stream
│       │   │   ├── jobs.py           # GET /jobs-list
│       │   │   └── results.py
│       │   └── result_listener.py    # background thread, Redis → MySQL
│       └── static/
│           ├── index.html            # upload dashboard
│           └── stream.html           # live stream detection
│
├── inference_service/
│   └── worker/
│       ├── main.py                   # FastAPI app + job worker thread
│       ├── inference.py              # model loading + run_inference
│       ├── stream_processor.py       # stream loop + frame aggregation
│       └── video_processor.py        # video frame extraction
│
├── uploads/
│   ├── images/                       # uploaded images
│   ├── videos/                       # uploaded videos
│   └── frames/                       # temp frames (auto-cleaned)
│
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Redis server running on `localhost:6379`
- MySQL server running on `localhost:3306`
- CUDA-capable GPU (optional — falls back to CPU)

### Installation

```bash
git clone https://github.com/BlackPrince2607/Verisight.git
cd Verisight
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in `api_service/`:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=verisight
DB_USER=your_user
DB_PASSWORD=your_password
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Running the Services

**Terminal 1 — API Service:**
```bash
cd api_service
python -m uvicorn app.api.main:app --port 8000 --reload
```

**Terminal 2 — Inference Service:**
```bash
cd inference_service/worker
python -m uvicorn main:app --port 8001 --reload
```

**Also required — Redis and MySQL must be running:**
```bash
sudo service redis-server start
sudo service mysql start
```

### Verify Everything Is Running

```
http://localhost:8000/health           →  API service status
http://localhost:8001/health           →  Inference service (shows device + model)
http://localhost:8000/static/index.html  →  Web dashboard
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload/` | Upload image or video for analysis |
| `GET` | `/status/{job_id}` | Poll job status and result |
| `GET` | `/jobs-list` | Fetch recent job history |
| `GET` | `/health` | API service health |
| `GET` | `/health/details` | Redis, DB, and worker status |
| `WS` | `/ws/stream` | WebSocket for live stream detection |
| `POST` | `/stream/start` | *(inference service)* Start stream analysis |
| `POST` | `/stream/stop` | *(inference service)* Stop stream analysis |

### WebSocket Protocol (`/ws/stream`)

**Client → Server:**
```json
{ "type": "start", "url": "https://youtube.com/watch?v=..." }
{ "type": "stop" }
```

**Server → Client:**
```json
{
  "type": "result",
  "frame": "data:image/jpeg;base64,...",
  "raw": { "label": "Fake", "confidence": 0.9821 },
  "smoothed": { "verdict": "FAKE", "confidence": 0.9634, "fake_ratio": 0.9634 },
  "frames_analyzed": 12
}
```

---

## Roadmap

- [ ] Docker + Docker Compose setup (single command deployment)
- [ ] Authentication + API key rate limiting
- [ ] Retry queue for failed jobs
- [ ] Model versioning (swap models without restarting)
- [ ] Monitoring + logging (Prometheus / Grafana)
- [ ] Fine-tune model on custom dataset
- [ ] Support for multiple concurrent streams
