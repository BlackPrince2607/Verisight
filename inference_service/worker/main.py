import time
import redis
import json
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

QUEUE_NAME = "analysis_jobs"
RESULT_QUEUE = "analysis_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

print("Loading model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

print("Model loaded on", DEVICE)


def run_inference(file_path):
    image = Image.open(file_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)
    label = model.config.id2label[predicted_class.item()]

    return label, float(confidence.item())


def start_inference_service():
    print("Inference Service Started. Waiting for jobs...")

    while True:
        job_data = redis_client.brpop(QUEUE_NAME)

        if job_data:
            _, message = job_data
            job_message = json.loads(message)

            job_id = job_message["job_id"]
            file_path = job_message["file_path"]

            try:
                start_time = time.time()

                label, confidence = run_inference(file_path)

                inference_time = int((time.time() - start_time) * 1000)

                result_message = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": label,
                    "confidence": confidence,
                    "inference_time_ms": inference_time
                }

            except Exception as e:
                result_message = {
                    "job_id": job_id,
                    "status": "failed",
                    "result": None,
                    "confidence": None,
                    "error": str(e)
                }

            redis_client.lpush(RESULT_QUEUE, json.dumps(result_message))


if __name__ == "__main__":
    start_inference_service()