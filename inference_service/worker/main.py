import time
import redis
import json
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, UnidentifiedImageError
import os

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

QUEUE_NAME = "analysis_jobs"
RESULT_QUEUE = "analysis_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"

print(f"Loading model on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("Model ready.")


def run_inference(file_path):
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    try:
        image = Image.open(file_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        raise ValueError(f"Could not load image at {file_path}: {e}")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    label = model.config.id2label[predicted_class.item()]

    return label, round(float(confidence.item()), 4)


def start_inference_service():
    print("Inference worker started. Waiting for jobs...")

    while True:
        try:
            job_data = redis_client.brpop(QUEUE_NAME, timeout=5)
            if not job_data:
                continue

            _, message = job_data
            job_message = json.loads(message)
            job_id = job_message["job_id"]
            file_path = job_message["file_path"]

            print(f"[JOB] Processing {job_id}")
            start_time = time.time()

            try:
                label, confidence = run_inference(file_path)
                inference_time = int((time.time() - start_time) * 1000)
                result_message = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": label,
                    "confidence": confidence,
                    "inference_time_ms": inference_time
                }
                print(f"[DONE] {job_id} → {label} ({confidence}) in {inference_time}ms")

            except Exception as e:
                print(f"[FAIL] Inference failed for {job_id}: {e}")
                result_message = {
                    "job_id": job_id,
                    "status": "failed",
                    "result": None,
                    "confidence": None,
                    "error": str(e)
                }

            redis_client.lpush(RESULT_QUEUE, json.dumps(result_message))

        except json.JSONDecodeError as e:
            print(f"[ERROR] Bad job JSON: {e}")
        except redis.RedisError as e:
            print(f"[ERROR] Redis connection issue: {e}. Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    start_inference_service()